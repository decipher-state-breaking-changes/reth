//! A hash map that can record the preimage of every change made to it, so the changes can be
//! undone.
//!
//! [`JournaledMap`] is the map the sparse trie keeps its nodes, values and branch masks in. While
//! it is recording, the first write to each key stores what that key held before — its value, or
//! the fact that it was absent — and every later write to the same key stores nothing. Restoring
//! the record puts every touched key back to what it held when recording began. A bulk clear
//! moves the whole map into the record instead of copying it; a bulk `retain` records only what
//! it removes.
//!
//! Recording is a property of the map, not of the call site: there is no way to change a
//! recording map's content that bypasses the record, because every mutating method is defined
//! here and reads go through `Deref`. That is what makes the record trustworthy across the
//! sparse trie's several hundred mutation sites without auditing each one.

use alloy_primitives::map::{self, HashMap};
use core::{
    hash::Hash,
    ops::{Deref, Index},
};

/// A hash map whose changes can be recorded and undone.
///
/// Reads go through `Deref` to the underlying map. Writes go through the methods defined here,
/// each of which records the preimage first when the map is recording. Cloning a recording map
/// yields a recording map with an empty record — a clone starts its own history — and equality
/// compares content only.
#[derive(Debug)]
pub struct JournaledMap<K, V> {
    map: HashMap<K, V>,
    journal: Option<MapJournal<K, V>>,
}

impl<K, V> Default for JournaledMap<K, V> {
    fn default() -> Self {
        Self { map: HashMap::default(), journal: None }
    }
}

impl<K, V> Deref for JournaledMap<K, V> {
    type Target = HashMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<K: Clone, V: Clone> Clone for JournaledMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            journal: self.journal.as_ref().map(|_| MapJournal::default()),
        }
    }
}

impl<K: Eq + Hash, V: PartialEq> PartialEq for JournaledMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl<K: Eq + Hash, V: Eq> Eq for JournaledMap<K, V> {}

impl<K: Eq + Hash, V> FromIterator<(K, V)> for JournaledMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self { map: HashMap::from_iter(iter), journal: None }
    }
}

impl<K: Eq + Hash, V> Index<&K> for JournaledMap<K, V> {
    type Output = V;

    fn index(&self, key: &K) -> &V {
        &self.map[key]
    }
}

impl<'a, K, V> IntoIterator for &'a JournaledMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = map::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl<K, V> JournaledMap<K, V> {
    /// Whether changes are currently being recorded.
    pub const fn is_recording(&self) -> bool {
        self.journal.is_some()
    }

    /// Starts recording. A no-op if already recording: the record in progress is kept, since it
    /// still describes the map as it was when that recording began.
    pub fn begin_recording(&mut self) {
        if self.journal.is_none() {
            self.journal = Some(MapJournal::default());
        }
    }
}

impl<K: Copy + Eq + Hash, V: Clone + PartialEq> JournaledMap<K, V> {
    /// Stops recording and returns the record, or `None` if the map was not recording.
    ///
    /// A key whose recorded preimage equals what it holds now is dropped from the record: a
    /// `get_mut` that changed nothing — the hash update visits every child of a dirty branch
    /// this way — would otherwise cost the record a copy of an unchanged node.
    pub fn take_journal(&mut self) -> Option<MapJournal<K, V>> {
        let mut journal = self.journal.take()?;
        if journal.whole.is_none() {
            let map = &self.map;
            journal.preimages.retain(|key, preimage| map.get(key) != preimage.as_ref());
        }
        Some(journal)
    }

    /// Reserves capacity; not a content change, so not recorded.
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
    }

    /// Shrinks capacity; not a content change, so not recorded.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.map.shrink_to(min_capacity);
    }

    /// Inserts a value, returning the one it displaced.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let previous = self.map.insert(key, value);
        if let Some(journal) = &mut self.journal {
            journal.record_with(key, || previous.clone());
        }
        previous
    }

    /// Removes a key, returning what it held.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let previous = self.map.remove(key);
        if let Some(journal) = &mut self.journal {
            journal.record_with(*key, || previous.clone());
        }
        previous
    }

    /// Mutable access to a value. Recorded as a write, since the caller may change it.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let value = self.map.get_mut(key)?;
        if let Some(journal) = &mut self.journal {
            journal.record_present(*key, value);
        }
        Some(value)
    }

    /// An entry in the map, for insert-or-update patterns.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.map.entry(key) {
            map::Entry::Occupied(entry) => {
                Entry::Occupied(OccupiedEntry { entry, journal: self.journal.as_mut() })
            }
            map::Entry::Vacant(entry) => {
                Entry::Vacant(VacantEntry { entry, journal: self.journal.as_mut() })
            }
        }
    }

    /// Keeps only the entries the predicate accepts, recording each one it removes.
    ///
    /// The predicate reads; it is not handed `&mut V`, so a retained entry cannot be changed
    /// behind the record.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        match &mut self.journal {
            Some(journal) if journal.whole.is_none() => {
                for (key, value) in self.map.extract_if(|key, value| !keep(key, value)) {
                    journal.record(key, Some(value));
                }
            }
            _ => self.map.retain(|key, value| keep(key, value)),
        }
    }

    /// Empties the map. While recording, the content moves into the record rather than being
    /// dropped, so the clear costs no copy and the restore costs no rebuild.
    pub fn clear(&mut self) {
        match &mut self.journal {
            Some(journal) if journal.whole.is_none() => {
                journal.whole = Some(core::mem::take(&mut self.map));
            }
            _ => self.map.clear(),
        }
    }

    /// Puts the map back to what it held when `journal` was begun.
    pub fn restore(&mut self, journal: MapJournal<K, V>) {
        if let Some(whole) = journal.whole {
            self.map = whole;
        }
        for (key, preimage) in journal.preimages {
            match preimage {
                Some(value) => self.map.insert(key, value),
                None => self.map.remove(&key),
            };
        }
    }
}

/// What a [`JournaledMap`] recorded between `begin_recording` and `take_journal`.
///
/// `preimages` holds, for each key written, what it held before its first write — `None` for a
/// key that was absent. `whole` holds the entire map as it stood before a bulk clear; once it is
/// set, later writes need no record because the restore replaces the map wholesale before
/// applying the preimages recorded ahead of the clear.
#[derive(Debug)]
pub struct MapJournal<K, V> {
    preimages: HashMap<K, Option<V>>,
    whole: Option<HashMap<K, V>>,
}

impl<K, V> Default for MapJournal<K, V> {
    fn default() -> Self {
        Self { preimages: HashMap::default(), whole: None }
    }
}

impl<K, V> MapJournal<K, V> {
    /// Whether the record describes no change at all.
    pub fn is_empty(&self) -> bool {
        self.preimages.is_empty() && self.whole.is_none()
    }

    /// Number of keys with a recorded preimage.
    pub fn len(&self) -> usize {
        self.preimages.len()
    }

    /// The preimages recorded, for accounting.
    pub fn preimages(&self) -> impl Iterator<Item = (&K, Option<&V>)> {
        self.preimages.iter().map(|(key, value)| (key, value.as_ref()))
    }

    /// The whole map captured by a bulk clear, if one happened.
    pub const fn whole(&self) -> Option<&HashMap<K, V>> {
        self.whole.as_ref()
    }

    /// Heap bytes the record holds, by hashbrown's sizing rule: the preimage table and the
    /// captured whole map charged their buckets, plus what each held value owns on the heap
    /// (`value_heap`). A heuristic — the rule restated, not the allocation read.
    pub fn allocated_bytes(&self, value_heap: impl Fn(&V) -> usize) -> usize {
        let mut bytes = hashbrown_table_bytes(
            self.preimages.capacity(),
            core::mem::size_of::<K>() + core::mem::size_of::<Option<V>>(),
        );
        bytes += self.preimages.values().flatten().map(&value_heap).sum::<usize>();
        if let Some(whole) = &self.whole {
            bytes += hashbrown_table_bytes(
                whole.capacity(),
                core::mem::size_of::<K>() + core::mem::size_of::<V>(),
            );
            bytes += whole.values().map(&value_heap).sum::<usize>();
        }
        bytes
    }
}

impl<K: Copy + Eq + Hash, V: Clone> MapJournal<K, V> {
    fn record(&mut self, key: K, previous: Option<V>) {
        if self.whole.is_some() {
            return;
        }
        self.preimages.entry(key).or_insert(previous);
    }

    /// Records `previous()` as the key's preimage on its first write only; a repeat write never
    /// evaluates it, so the copy is paid once per key, not once per write.
    fn record_with(&mut self, key: K, previous: impl FnOnce() -> Option<V>) {
        if self.whole.is_some() {
            return;
        }
        self.preimages.entry(key).or_insert_with(previous);
    }

    fn record_present(&mut self, key: K, value: &V) {
        if self.whole.is_some() {
            return;
        }
        self.preimages.entry(key).or_insert_with(|| Some(value.clone()));
    }
}

/// Bytes a hashbrown table with this reported capacity holds: its power-of-two bucket count
/// times one entry and one control byte. hashbrown's `capacity_to_buckets`, restated.
pub(crate) const fn hashbrown_table_bytes(capacity: usize, entry_bytes: usize) -> usize {
    let buckets = match capacity {
        0 => return 0,
        1..=3 => 4,
        4..=7 => 8,
        capacity => (capacity * 8 / 7).next_power_of_two(),
    };
    buckets * (entry_bytes + 1)
}

/// A view into a single entry of a [`JournaledMap`].
#[derive(Debug)]
pub enum Entry<'a, K, V> {
    /// The key is present.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The key is absent.
    Vacant(VacantEntry<'a, K, V>),
}

/// An occupied entry. Reading records nothing; any write records the value first.
#[derive(Debug)]
pub struct OccupiedEntry<'a, K, V> {
    entry: map::OccupiedEntry<'a, K, V>,
    journal: Option<&'a mut MapJournal<K, V>>,
}

impl<'a, K: Copy + Eq + Hash, V: Clone> OccupiedEntry<'a, K, V> {
    /// The entry's key.
    pub fn key(&self) -> &K {
        self.entry.key()
    }

    /// The value, read-only.
    pub fn get(&self) -> &V {
        self.entry.get()
    }

    fn record(&mut self) {
        if let Some(journal) = &mut self.journal {
            journal.record_present(*self.entry.key(), self.entry.get());
        }
    }

    /// The value, mutably.
    pub fn get_mut(&mut self) -> &mut V {
        self.record();
        self.entry.get_mut()
    }

    /// The value, mutably, for the map's lifetime.
    pub fn into_mut(mut self) -> &'a mut V {
        self.record();
        self.entry.into_mut()
    }

    /// Replaces the value, returning the old one.
    pub fn insert(&mut self, value: V) -> V {
        self.record();
        self.entry.insert(value)
    }

    /// Removes the entry, returning its value.
    pub fn remove(mut self) -> V {
        self.record();
        self.entry.remove()
    }
}

/// A vacant entry. Inserting records the key as having been absent.
#[derive(Debug)]
pub struct VacantEntry<'a, K, V> {
    entry: map::VacantEntry<'a, K, V>,
    journal: Option<&'a mut MapJournal<K, V>>,
}

impl<'a, K: Copy + Eq + Hash, V: Clone> VacantEntry<'a, K, V> {
    /// The entry's key.
    pub fn key(&self) -> &K {
        self.entry.key()
    }

    /// Inserts a value, returning a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        if let Some(journal) = self.journal {
            journal.record(*self.entry.key(), None);
        }
        self.entry.insert(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_of(pairs: &[(u8, &str)]) -> JournaledMap<u8, Vec<u8>> {
        pairs.iter().map(|(k, v)| (*k, v.as_bytes().to_vec())).collect()
    }

    fn content(map: &JournaledMap<u8, Vec<u8>>) -> Vec<(u8, Vec<u8>)> {
        let mut pairs: Vec<_> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn a_map_that_is_not_recording_records_nothing() {
        let mut map = map_of(&[(1, "a")]);
        map.insert(2, b"b".to_vec());
        map.remove(&1);
        assert!(map.take_journal().is_none());
        assert_eq!(content(&map), vec![(2, b"b".to_vec())]);
    }

    #[test]
    fn every_kind_of_write_is_undone_to_the_state_recording_began_in() {
        let mut map = map_of(&[(1, "a"), (2, "b"), (3, "c"), (4, "d")]);
        let before = map.clone();
        map.begin_recording();

        map.insert(1, b"a2".to_vec());
        map.insert(1, b"a3".to_vec());
        map.remove(&2);
        map.get_mut(&3).unwrap().push(b'!');
        map.insert(5, b"e".to_vec());
        match map.entry(4) {
            Entry::Occupied(mut e) => {
                e.insert(b"d2".to_vec());
            }
            Entry::Vacant(_) => unreachable!(),
        }
        match map.entry(6) {
            Entry::Vacant(e) => {
                e.insert(b"f".to_vec());
            }
            Entry::Occupied(_) => unreachable!(),
        }
        match map.entry(5) {
            Entry::Occupied(e) => {
                e.remove();
            }
            Entry::Vacant(_) => unreachable!(),
        }
        map.retain(|k, _| *k != 6 && *k != 3);

        let journal = map.take_journal().unwrap();
        assert!(!map.is_recording());
        assert_eq!(
            journal.len(),
            4,
            "keys 1 2 3 4 changed; 5 and 6 were inserted and removed again, so their record is dropped"
        );
        map.restore(journal);
        assert_eq!(map, before);
    }

    #[test]
    fn a_clear_moves_the_map_into_the_record_and_writes_after_it_leave_no_trace() {
        let mut map = map_of(&[(1, "a"), (2, "b")]);
        let before = map.clone();
        map.begin_recording();
        map.insert(1, b"a2".to_vec());
        map.clear();
        assert!(map.is_empty());
        map.insert(1, b"z".to_vec());
        map.insert(9, b"y".to_vec());
        map.clear();

        let journal = map.take_journal().unwrap();
        assert_eq!(journal.len(), 1, "only the write ahead of the first clear is a preimage");
        assert_eq!(journal.whole().map(|m| m.len()), Some(2));
        map.restore(journal);
        assert_eq!(map, before);
    }

    #[test]
    fn a_clone_of_a_recording_map_starts_its_own_record() {
        let mut map = map_of(&[(1, "a")]);
        map.begin_recording();
        map.insert(2, b"b".to_vec());
        let mut copy = map.clone();
        assert!(copy.is_recording());
        copy.insert(3, b"c".to_vec());

        let copy_journal = copy.take_journal().unwrap();
        assert_eq!(copy_journal.len(), 1, "the copy did not inherit the parent's record");
        copy.restore(copy_journal);
        assert_eq!(copy, map, "the copy undoes to the parent's current content, not its origin");

        let journal = map.take_journal().unwrap();
        map.restore(journal);
        assert_eq!(content(&map), vec![(1, b"a".to_vec())]);
    }

    #[test]
    fn beginning_twice_keeps_the_record_in_progress() {
        let mut map = map_of(&[(1, "a")]);
        map.begin_recording();
        map.insert(1, b"a2".to_vec());
        map.begin_recording();
        map.insert(1, b"a3".to_vec());
        let journal = map.take_journal().unwrap();
        map.restore(journal);
        assert_eq!(content(&map), vec![(1, b"a".to_vec())]);
    }
}
