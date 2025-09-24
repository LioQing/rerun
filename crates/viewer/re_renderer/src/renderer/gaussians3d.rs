use std::sync::OnceLock;

use crate::{
    DebugLabel, DrawPhase, DrawableCollector, GpuRenderPipelinePoolAccessor, MsaaMode,
    RenderContext, ViewBuilder, ViewTargetSetup,
    renderer::{
        DrawData, DrawDataDrawable, DrawError, DrawInstruction, DrawableCollectionViewInfo,
        Renderer,
    },
};
use wgpu_3dgs_viewer as gs;

type GaussianPod = gs::DefaultGaussianPod;
type Preprocessor = gs::Preprocessor<GaussianPod, ()>;
type RadixSorter = gs::RadixSorter<()>;
type GsRenderer = gs::Renderer<GaussianPod, ()>;

#[allow(dead_code)]
pub struct Gaussians3DDrawData {
    camera: gs::CameraBuffer,
    model_transform: gs::core::ModelTransformBuffer,
    gaussian_transform: gs::core::GaussianTransformBuffer,
    gaussians: gs::core::GaussiansBuffer<GaussianPod>,
    indirect_args: gs::IndirectArgsBuffer,
    radix_sort_indirect_args: gs::RadixSortIndirectArgsBuffer,
    indirect_indices: gs::IndirectIndicesBuffer,
    gaussians_depth_buffer: gs::GaussiansDepthBuffer,

    preprocessor_bind_group: wgpu::BindGroup,
    radix_sorter_bind_group: gs::RadixSorterBindGroups,
    renderer_bind_group: wgpu::BindGroup,
}

impl DrawData for Gaussians3DDrawData {
    type Renderer = Gaussians3DRenderer;

    fn collect_drawables(
        &self,
        _view_info: &DrawableCollectionViewInfo,
        collector: &mut DrawableCollector<'_>,
    ) {
        collector.add_drawable_for_phase(
            DrawPhase::Transparent,
            DrawDataDrawable {
                distance_sort_key: 0.0,
                draw_data_payload: 0,
            },
        );
    }
}

impl Gaussians3DDrawData {
    pub fn new(device: &wgpu::Device, gaussians: &gs::core::Gaussians) -> Self {
        let camera = gs::CameraBuffer::new(device);
        let model_transform = gs::core::ModelTransformBuffer::new(device);
        let gaussian_transform = gs::core::GaussianTransformBuffer::new(device);
        let gaussians = gs::core::GaussiansBuffer::new(device, &gaussians.gaussians);
        let indirect_args = gs::IndirectArgsBuffer::new(device);
        let radix_sort_indirect_args = gs::RadixSortIndirectArgsBuffer::new(device);
        let indirect_indices = gs::IndirectIndicesBuffer::new(device, gaussians.len() as u32);
        let gaussians_depth_buffer = gs::GaussiansDepthBuffer::new(device, gaussians.len() as u32);

        let (preprocessor, radix_sorter, renderer) = pipelines(device);

        let preprocessor_bind_group = preprocessor.create_bind_group(
            device,
            &camera,
            &model_transform,
            &gaussian_transform,
            &gaussians,
            &indirect_args,
            &radix_sort_indirect_args,
            &indirect_indices,
            &gaussians_depth_buffer,
        );

        let radix_sorter_bind_group =
            radix_sorter.create_bind_groups(device, &gaussians_depth_buffer, &indirect_indices);

        let renderer_bind_group = renderer.create_bind_group(
            device,
            &camera,
            &model_transform,
            &gaussian_transform,
            &gaussians,
            &indirect_indices,
        );

        Self {
            camera,
            model_transform,
            gaussian_transform,
            gaussians,
            indirect_args,
            radix_sort_indirect_args,
            indirect_indices,
            gaussians_depth_buffer,

            preprocessor_bind_group,
            radix_sorter_bind_group,
            renderer_bind_group,
        }
    }
}

// Instead of using the render context's gpu resource's [`GpuBindGroupLayoutPool::get_or_create`],
// We are going to use static variables to memoize the pipelines.

static PREPROCESSOR: OnceLock<Preprocessor> = OnceLock::new();
static RADIX_SORTER: OnceLock<RadixSorter> = OnceLock::new();
static RENDERER: OnceLock<GsRenderer> = OnceLock::new();

fn pipelines(
    device: &wgpu::Device,
) -> (
    &'static Preprocessor,
    &'static RadixSorter,
    &'static GsRenderer,
) {
    (
        PREPROCESSOR.get_or_init(|| {
            Preprocessor::new_without_bind_group(device).expect("gaussians preprocessor")
        }),
        RADIX_SORTER.get_or_init(|| RadixSorter::new_without_bind_groups(device)),
        RENDERER.get_or_init(|| {
            GsRenderer::new_without_bind_group(
                device,
                ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                Some(ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE_NO_WRITE),
            )
            .expect("gaussians renderer")
        }),
    )
}

pub struct Gaussians3DRenderer {
    preprocessor: &'static Preprocessor,
    radix_sorter: &'static RadixSorter,
    renderer: &'static GsRenderer,
}

impl Renderer for Gaussians3DRenderer {
    type RendererDrawData = Gaussians3DDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        let (preprocessor, radix_sorter, renderer) = pipelines(&ctx.device);

        Self {
            preprocessor,
            radix_sorter,
            renderer,
        }
    }

    fn cleanup(
        &self,
        ctx: &RenderContext,
        setup: &ViewTargetSetup,
        _render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        encoder: &mut wgpu::CommandEncoder,
        draw_instructions: &[DrawInstruction<'_, Self::RendererDrawData>],
    ) -> Result<(), DrawError> {
        let Some(DrawInstruction { draw_data, .. }) = draw_instructions.first() else {
            return Ok(());
        };

        let queue = &ctx.queue;

        struct Camera {
            pub proj: glam::Mat4,
            pub view: glam::Mat4,
        }

        impl gs::CameraTrait for Camera {
            fn projection(&self, _aspect_ratio: f32) -> glam::Mat4 {
                self.proj
            }
            fn view(&self) -> glam::Mat4 {
                self.view
            }
        }

        draw_data.camera.update_with_pod(
            queue,
            &gs::CameraPod::new(
                &Camera {
                    proj: setup.projection_from_view,
                    view: setup.view_from_world,
                },
                glam::UVec2::from_array(setup.resolution_in_pixel),
            ),
        );

        self.preprocessor.preprocess(
            encoder,
            &draw_data.preprocessor_bind_group,
            draw_data.gaussians.len() as u32,
        );

        self.radix_sorter.sort(
            encoder,
            &draw_data.radix_sorter_bind_group,
            &draw_data.radix_sort_indirect_args,
        );

        {
            let needs_msaa_resolve = ctx.render_config().msaa_mode != MsaaMode::Off;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: DebugLabel::from(format!("{} - main gaussians pass", setup.name)).get(),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &setup.main_target_msaa.default_view,
                    resolve_target: needs_msaa_resolve
                        .then_some(&setup.main_target_resolved.default_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &setup.depth_buffer.default_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.renderer.render_with_pass(
                &mut pass,
                &draw_data.renderer_bind_group,
                &draw_data.indirect_args,
            );
        }

        Ok(())
    }

    fn draw(
        &self,
        _render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        _phase: DrawPhase,
        _pass: &mut wgpu::RenderPass<'_>,
        _draw_instructions: &[DrawInstruction<'_, Self::RendererDrawData>],
    ) -> Result<(), DrawError> {
        Ok(())
    }
}
