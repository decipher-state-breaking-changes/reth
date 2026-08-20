//! The background state population a witness is proved against.
//!
//! A witness's size is decided by two things: which keys a block touches, and how deep those keys
//! sit in the tree. The first is recorded in the corpus and is exact. The second is a property of
//! the *whole* state — a key is only as deep as it needs to be to separate from every other key —
//! and the corpus does not carry the whole state.
//!
//! Materialising mainnet's state under three different commitment schemes is not affordable here,
//! and it is also not necessary. Every scheme in this study derives its tree position by hashing,
//! so the positions of the keys this study does not name are uniform over the key space. A uniform
//! population of a known size is fully described by that size. This module therefore models the
//! background as `size` keys drawn uniformly, and samples it *lazily*: the count under a prefix is
//! produced by a deterministic pseudo-random function of that prefix, so the same prefix always
//! yields the same count, sibling counts always sum to their parent's, and no key is ever stored.
//!
//! Two properties make this an honest substitute rather than a convenience:
//!
//! 1. The population size is measured, not assumed. It comes from the same node whose blocks the
//!    corpus records — see the `state-size` provenance recorded in the run report.
//! 2. Witness size depends on the population only through `log2(size)`, so a factor-of-two error in
//!    the size moves every path by one level. The study sweeps the size for exactly this reason and
//!    reports the sensitivity rather than asking to be believed.
//!
//! What this does *not* model is any correlation between real keys' positions. There is none to
//! model for hash-derived positions, which is why the schemes are compared on this basis and why
//! the MPT arm — whose real witness bytes are known from the corpus — is replayed through the same
//! machinery as a calibration.

use crate::keys::Prefix;

/// A uniformly distributed background state of a known size, sampled lazily.
#[derive(Debug, Clone)]
pub struct BackgroundPopulation {
    /// How many keys the modelled state holds.
    size: u64,
    /// Domain separator, so two populations in one run (accounts vs storage, or a sensitivity arm)
    /// never share a draw.
    seed: [u8; 16],
}

impl BackgroundPopulation {
    /// A population of `size` keys, sampled under `label`.
    pub fn new(size: u64, label: &str) -> Self {
        let digest = blake3::hash(label.as_bytes());
        let mut seed = [0u8; 16];
        seed.copy_from_slice(&digest.as_bytes()[..16]);
        Self { size, seed }
    }

    /// The modelled state size.
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// How many background keys lie under `prefix`.
    ///
    /// Walks from the root splitting the count at each level, so the answer is consistent with
    /// every ancestor and with the sibling that shares each split.
    pub fn count_under(&self, prefix: Prefix) -> u64 {
        let mut count = self.size;
        for depth in 1..=prefix.len() {
            let parent = prefix.truncated(depth - 1);
            let left = self.split_left(parent, count);
            count = if prefix.bit(depth - 1) { count - left } else { left };
            if count == 0 {
                return 0
            }
        }
        count
    }

    /// Walks the path `key` takes from the root, reporting what the background leaves beside it.
    ///
    /// Returns the sibling count at every level down to the depth at which `key` is alone —
    /// the depth its own node can sit at, since a node with no other key beneath it needs no
    /// further branching. One split is drawn per level, and the split yields both children at
    /// once, so the whole walk costs one draw per level rather than one per query.
    pub fn descend(&self, key: &[u8; 32], max_depth: u32) -> Descent {
        let mut siblings = Vec::with_capacity(max_depth as usize);
        let mut count = self.size;
        let mut prefix = Prefix::root();
        let mut depth = 0;
        while depth < max_depth {
            if count == 0 {
                break
            }
            let left = self.split_left(prefix, count);
            let right = count - left;
            let bit = bit_of(key, depth);
            let (mine, theirs) = if bit { (right, left) } else { (left, right) };
            siblings.push(theirs);
            count = mine;
            prefix = prefix.pushed(bit);
            depth += 1;
        }
        Descent { alone_at: depth, siblings }
    }

    /// Splits `count` keys under `parent` between its two children, returning the left share.
    ///
    /// Deterministic in `parent`, so the same node always splits the same way and the two children
    /// always sum back to the parent.
    fn split_left(&self, parent: Prefix, count: u64) -> u64 {
        if count == 0 {
            return 0
        }
        if count == 1 {
            return u64::from(!self.uniform_bit(parent));
        }
        let u = self.uniform_unit(parent);
        binomial_half(count, u)
    }

    /// A uniform in `[0, 1)` keyed by `prefix`.
    fn uniform_unit(&self, prefix: Prefix) -> f64 {
        let raw = self.draw(prefix);
        // 53 bits is the whole mantissa; taking more would not change the value.
        ((raw >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// A single uniform bit keyed by `prefix`.
    fn uniform_bit(&self, prefix: Prefix) -> bool {
        self.draw(prefix) & 1 == 1
    }

    fn draw(&self, prefix: Prefix) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed);
        hasher.update(&prefix.bytes());
        hasher.update(&[prefix.len() as u8]);
        let out = hasher.finalize();
        u64::from_le_bytes(out.as_bytes()[..8].try_into().expect("8 bytes"))
    }
}

/// Draws `Binomial(n, 1/2)` from one uniform `u`.
///
/// Exact by inverse CDF while `n` is small enough for the CDF to be worth walking, and a normal
/// approximation above that. The crossover is chosen where the approximation's error is already far
/// below one key; at the depths where `n` is large, both children are near `n/2` regardless, and
/// the split only starts to matter for the answer once `n` is small — which is the exact branch.
fn binomial_half(n: u64, u: f64) -> u64 {
    const EXACT_LIMIT: u64 = 64;
    if n <= EXACT_LIMIT {
        return binomial_half_exact(n as u32, u) as u64
    }
    let mean = n as f64 / 2.0;
    let sd = (n as f64).sqrt() / 2.0;
    let z = inverse_standard_normal(u);
    let value = (mean + sd * z).round();
    value.clamp(0.0, n as f64) as u64
}

/// Inverse CDF of `Binomial(n, 1/2)`, walking the pmf up from `k = 0`.
fn binomial_half_exact(n: u32, u: f64) -> u32 {
    // pmf(0) = 2^-n; the whole row is scaled by that, so accumulate in the same scale.
    let mut pmf = (0.5f64).powi(n as i32);
    let mut cdf = pmf;
    let mut k = 0u32;
    while k < n && cdf <= u {
        // pmf(k+1) = pmf(k) * (n - k) / (k + 1)
        pmf = pmf * f64::from(n - k) / f64::from(k + 1);
        cdf += pmf;
        k += 1;
    }
    k
}

/// Acklam's rational approximation to the standard normal quantile.
///
/// Accurate to about 1.15e-9 in absolute value over the whole range, which is far tighter than the
/// nearest-integer rounding this feeds.
///
/// Left in the published Horner form. Folding the multiply-adds would be marginally more accurate
/// and would also make the coefficients unrecognisable against the source they are quoted from,
/// which matters more for a constant table nobody can check by reading.
#[expect(clippy::suboptimal_flops)]
fn inverse_standard_normal(u: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] =
        [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
    const LOW: f64 = 0.02425;

    let p = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    if p < LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
    if p > 1.0 - LOW {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q /
        (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

/// What a background population leaves beside one key's path.
#[derive(Debug, Clone)]
pub struct Descent {
    /// Depth at which no background key shares the path any longer.
    pub alone_at: u32,
    /// Background keys under the sibling at each level, index 0 being depth 1.
    pub siblings: Vec<u64>,
}

impl Descent {
    /// Background keys under the sibling of the node at `depth`, counting the root as depth 0.
    pub fn sibling_count(&self, depth: u32) -> u64 {
        self.siblings.get((depth - 1) as usize).copied().unwrap_or(0)
    }
}

/// Bit `index` of `key`, most significant bit of byte zero first.
const fn bit_of(key: &[u8; 32], index: u32) -> bool {
    key[(index / 8) as usize] >> (7 - index % 8) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::Prefix;

    #[test]
    fn children_sum_to_their_parent() {
        let pop = BackgroundPopulation::new(1_000_000_000, "test");
        let mut checked = 0;
        for depth in 0..24u32 {
            let parent = Prefix::from_bits(0x5a5a_5a5a_5a5a_5a5a, depth);
            let total = pop.count_under(parent);
            let left = pop.count_under(parent.pushed(false));
            let right = pop.count_under(parent.pushed(true));
            assert_eq!(left + right, total, "split lost keys at depth {depth}");
            checked += 1;
        }
        assert_eq!(checked, 24);
    }

    #[test]
    fn a_prefix_always_answers_the_same_way() {
        let pop = BackgroundPopulation::new(412_044_818, "accounts");
        let prefix = Prefix::from_bits(0xdead_beef_dead_beef, 30);
        assert_eq!(pop.count_under(prefix), pop.count_under(prefix));
    }

    #[test]
    fn paths_bottom_out_near_log2_of_the_population() {
        // A population of 2^30 keys should leave a random path alone within a few levels of 30.
        let pop = BackgroundPopulation::new(1 << 30, "depth");
        let mut depths = Vec::new();
        for i in 0..256u64 {
            let bits = u64::from_le_bytes(
                blake3::hash(&i.to_le_bytes()).as_bytes()[..8].try_into().expect("8 bytes"),
            );
            let mut depth = 0u32;
            while depth < 60 && pop.count_under(Prefix::from_bits(bits, depth)) > 1 {
                depth += 1;
            }
            depths.push(depth);
        }
        let mean = depths.iter().map(|d| f64::from(*d)).sum::<f64>() / depths.len() as f64;
        assert!(mean > 26.0 && mean < 34.0, "mean terminal depth {mean} is not near log2(2^30)");
    }

    #[test]
    fn an_empty_population_has_no_keys_anywhere() {
        let pop = BackgroundPopulation::new(0, "empty");
        assert_eq!(pop.count_under(Prefix::from_bits(0, 0)), 0);
        assert_eq!(pop.count_under(Prefix::from_bits(0xff, 8)), 0);
    }
}
