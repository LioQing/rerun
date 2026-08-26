"""Generated gRPC/protobuf stubs for the Rerun viewer's `ViewerControlService`.

These modules are generated from the Rerun repo's
`crates/store/re_protos/proto/rerun/v1alpha1/{viewer,common}.proto` using:

    protoc -I crates/store/re_protos/proto \
        --python_out=viewer_proto/v1alpha1 \
        --grpclib_python_out=viewer_proto/v1alpha1 \
        crates/store/re_protos/proto/rerun/v1alpha1/viewer.proto \
        crates/store/re_protos/proto/rerun/v1alpha1/common.proto

The only edit made after generation is rewiring the generated
`rerun.v1alpha1.*` imports to package-relative imports so they do not collide
with the `rerun` SDK package. Regenerate with the same commands and re-apply
that import rewrite.
"""
