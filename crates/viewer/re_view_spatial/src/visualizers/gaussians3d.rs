use re_log_types::EntityPath;
use re_renderer;
use re_renderer::renderer::Gaussians3DDrawData;
use re_types::{
    Archetype,
    archetypes::Gaussians3D,
    components::{Color, ShowLabels},
    external::re_types_core,
};
use re_view::{DataResultQuery, RangeResultsExt};
use re_viewer_context::{
    self, IdentifiedViewSystem, MaybeVisualizableEntities, QueryContext,
    TypedComponentFallbackProvider, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, VisualizableEntities,
    VisualizableFilterContext, VisualizerQueryInfo, VisualizerSystem,
};
use wgpu_3dgs_viewer::core::{Gaussian, Gaussians};

use crate::visualizers::filter_visualizable_3d_entities;

#[derive(Default)]
pub struct Gaussians3DVisualizer {
    pub gaussians: Vec<(EntityPath, Gaussians)>,
}

impl IdentifiedViewSystem for Gaussians3DVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "Gaussians3D".into()
    }
}

impl VisualizerSystem for Gaussians3DVisualizer {
    fn visualizer_query_info(&self) -> VisualizerQueryInfo {
        VisualizerQueryInfo::from_archetype::<Gaussians3D>()
    }

    fn filter_visualizable_entities(
        &self,
        entities: MaybeVisualizableEntities,
        context: &dyn VisualizableFilterContext,
    ) -> VisualizableEntities {
        re_tracing::profile_function!();
        filter_visualizable_3d_entities(entities, context)
    }

    /// Populates the visualizer with data from the store.
    fn execute(
        &mut self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        _context_systems: &ViewContextCollection,
    ) -> Result<Vec<re_renderer::QueueableDrawData>, ViewSystemExecutionError> {
        for data_result in query.iter_visible_data_results(Self::identifier()) {
            let results = data_result.query_components_with_history(
                ctx,
                query,
                Gaussians3D::required_components()
                    .iter()
                    .chain(std::iter::once(&Gaussians3D::descriptor_sh()))
                    .collect::<Vec<_>>(),
            );

            let rots = results.iter_as(query.timeline, Gaussians3D::descriptor_rots());
            let poss = results.iter_as(query.timeline, Gaussians3D::descriptor_poss());
            let colors = results.iter_as(query.timeline, Gaussians3D::descriptor_colors());
            let shs = results.iter_as(query.timeline, Gaussians3D::descriptor_sh());
            let scales = results.iter_as(query.timeline, Gaussians3D::descriptor_scales());

            let shs_vec = shs.slice::<[f32; 45]>().collect::<Vec<_>>();

            let mut gaussians_vec = Vec::new();
            let shs_placeholder = (
                (re_log_types::TimeInt::MIN, re_types_core::RowId::ZERO),
                [].as_slice(),
            );
            for ((_, rot), (_, pos), (_, color), (_, sh), (_, scale)) in itertools::izip!(
                rots.slice::<[f32; 4]>(),
                poss.slice::<[f32; 3]>(),
                colors.slice::<u32>(),
                match shs_vec.len() {
                    0 => itertools::Either::Left(std::iter::repeat(&shs_placeholder)),
                    _ => itertools::Either::Right(shs_vec.iter()),
                },
                scales.slice::<[f32; 3]>()
            ) {
                for (rot, pos, color, sh, scale) in itertools::izip!(
                    rot.iter(),
                    pos.iter(),
                    color.iter(),
                    sh.iter().chain(std::iter::repeat(&[0.0; 45])),
                    scale.iter()
                ) {
                    let gaussian = Gaussian {
                        rot: glam::Quat::from_array(*rot),
                        pos: glam::Vec3::from_array(*pos),
                        color: glam::U8Vec4::from(Color::from_u32(*color).to_array()),
                        scale: glam::Vec3::from_array(*scale),
                        sh: sh
                            .chunks(3)
                            .map(|c| glam::Vec3::from_slice(c))
                            .take(15)
                            .collect::<Vec<_>>()
                            .try_into()
                            .expect("slice with incorrect length"),
                    };

                    gaussians_vec.push(gaussian);
                }
            }

            if gaussians_vec.is_empty() {
                re_log::warn!("Logged Gaussians3D has no instance");
                continue;
            }

            if let Some((_, gaussians)) = self
                .gaussians
                .iter_mut()
                .find(|(path, _)| *path == data_result.entity_path)
            {
                *gaussians = Gaussians {
                    gaussians: gaussians_vec,
                };
            } else {
                self.gaussians.push((
                    data_result.entity_path.clone(),
                    Gaussians {
                        gaussians: gaussians_vec,
                    },
                ));
            }
        }

        let draw_data = self
            .gaussians
            .iter()
            .map(|(_, g)| Gaussians3DDrawData::new(&ctx.viewer_ctx.render_ctx().device, g).into())
            .collect::<Vec<_>>();

        Ok(draw_data)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn fallback_provider(&self) -> &dyn re_viewer_context::ComponentFallbackProvider {
        &Fallback
    }
}

struct Fallback;

impl TypedComponentFallbackProvider<ShowLabels> for Fallback {
    fn fallback_for(&self, ctx: &QueryContext<'_>) -> ShowLabels {
        super::utilities::show_labels_fallback(
            ctx,
            &Gaussians3D::descriptor_scales(),
            &Gaussians3D::descriptor_labels(),
        )
    }
}

re_viewer_context::impl_component_fallback_provider!(Fallback => [ShowLabels]);
