fn main() {
    tonic_build::compile_protos("proto/partial_stateless_sidecar.proto")
        .unwrap_or_else(|err| panic!("failed to compile partial-stateless sidecar proto: {err:?}"));
}
