//! A round on a curved face is certified from the face's partials over the feet's boxes.
//! A tilted circular loft's side is a rational ruled face (an elliptic cylinder), and the
//! round on its rim used to subdivide until those enclosures shrank. This case is the
//! small one from the report: two circles of radius 4, 10 apart, the top centre moved
//! 2.1256 in x, the bottom rim rounded at 1.5. The straight loft (the same circles, one
//! axis) is the control: its side is a cylinder and the round is exact.
//!
//! The volumes are the loft's `160 π` less the round. They were measured on this kernel
//! with the definition round (`fillet_round_as_definition`) and checked against the
//! straight loft's closed form.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)] // test helpers

use std::time::Instant;

use solidcore::blend::fillet_executive::api::fillet_edges;
use solidcore::curve::curve::Curve;
use solidcore::curve::ellipse::Ellipse;
use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::geom::axis2_placement::Axis2Placement;
use solidcore::geom::extent1d::Extent1d;
use solidcore::mass::integrator::properties::{brep_volume, PropertyAttributes};
use solidcore::math::{Point3, Vec3};
use solidcore::query::api_queries::{brep_is_manifold_solid, edge_closest_point};
use solidcore::surface::surface::SurfaceKind;
use solidcore::sweep::api_surfaces::create_skin_primitive;
use solidcore::topology::face::p2 as face;
use solidcore::{Brep, EdgeId, Kernel};

/// SolidCraft's fillet flags that choose the definition round (`crates/smlib/src/core.rs`).
fn kernel() -> Kernel {
    let mut k = Kernel::default();
    k.fixes.fillet_gn_section_generator = true;
    k.fixes.fillet_round_as_definition = true;
    k.fixes.fillet_exact_arc_sections = true;
    k.fixes.fillet_exact_arc_spans = true;
    k.fixes.fillet_surface_plane_curve_on_plane_equation = true;
    k
}

/// A capped ruled loft between two circles of radius 4. `shift` moves the top centre in x.
fn loft(k: &mut Kernel, shift: f64) -> Brep {
    let mut curves = Vec::new();
    for (z, x) in [(0.0, 0.0), (10.0, shift)] {
        let frame = Axis2Placement::new(Vec3::new(x, 0.0, z), Vec3::X, Vec3::Y);
        curves.push(Curve::from(
            Ellipse::create_canonical_circle(
                k,
                &frame,
                4.0,
                Some(&Extent1d::from_min_max(0.0, 360.0)),
                None,
            )
            .unwrap(),
        ));
    }
    let refs: Vec<_> = curves.iter().map(Some).collect();
    let mut part = create_skin_primitive(k, &[1, 1], &refs, 1, true)
        .unwrap()
        .brep;
    part.refresh_cached_topology();
    part
}

fn volume(k: &Kernel, b: &mut Brep) -> f64 {
    let store = AttributeStore::default();
    let attrs = PropertyAttributes {
        store: &store,
        key_of: &|_| ObjectKey(0),
    };
    brep_volume(k, b, 1e-6, &attrs).unwrap()
}

fn nearest_edge(k: &Kernel, b: &Brep, p: Point3) -> EdgeId {
    let mut best = (1.0_f64, None);
    for &e in b.edge_list() {
        if let Ok(c) = edge_closest_point(k, b, e, p) {
            if c.closest.distance < best.0 {
                best = (c.closest.distance, Some(e));
            }
        }
    }
    best.1.unwrap_or_else(|| panic!("no edge near {p:?}"))
}

/// The round faces' certified bounds, and the largest net (control points along each way).
fn round_nets(b: &Brep) -> (usize, f64, usize) {
    let mut rounds = 0;
    let mut bound = 0.0_f64;
    let mut points = 0_usize;
    for f in b.faces().ids() {
        let Ok(Some(srf)) = face::surface(b, f) else {
            continue;
        };
        if let Some(blend) = srf.procedural_blend() {
            rounds += 1;
            bound = bound.max(blend.bound());
            if let SurfaceKind::ProceduralBlend(p) = srf.kind() {
                if let Ok(net) = p.net().control_point_net() {
                    points = points.max(net.points.len());
                }
            }
        }
    }
    (rounds, bound, points)
}

fn fillet_rim(shift: f64) -> (f64, usize, f64, usize, f64) {
    let mut k = kernel();
    let mut body = loft(&mut k, shift);
    let edge = nearest_edge(&k, &body, Point3::new(4.0, 0.0, 0.0));
    let started = Instant::now();
    fillet_edges(&mut k, &mut body, &[edge], 1.5, 1, 1, 1.0).unwrap();
    let secs = started.elapsed().as_secs_f64();
    body.refresh_cached_topology();
    assert!(brep_is_manifold_solid(&body).unwrap());
    let vol = volume(&k, &mut body);
    let (rounds, bound, points) = round_nets(&body);
    if std::env::var_os("FILLET_BENCH").is_some() {
        eprintln!(
            "FILLET_BENCH shift={shift} secs={secs:.3} volume={vol:.6} rounds={rounds} bound={bound:.3e} net_points={points}"
        );
    }
    #[cfg(feature = "profile")]
    {
        let mut out = std::fs::File::create(format!("/tmp/fillet-profile-{shift}.jsonl")).unwrap();
        k.take_profile().write_jsonl(&mut out).unwrap();
    }
    (secs, rounds, bound, points, vol)
}

/// Volume of the straight loft after the outside round: the cylinder `π R² H` less the
/// corner the round replaces. That corner, in a radial section, is the square `r × r`
/// minus the quarter disk. Pappus takes its area times the circle through its centroid.
/// The centroid of a quarter disk sits `4r / (3π)` from the disk's centre along each axis.
fn straight_round_volume() -> f64 {
    let (big_r, r, height) = (4.0, 1.5, 10.0);
    let along = 4.0 / (3.0 * std::f64::consts::PI);
    let x_square = big_r - r / 2.0;
    let x_disk = (big_r - r) + along * r;
    let rest = 1.0 - std::f64::consts::FRAC_PI_4;
    let x_removed = (x_square - std::f64::consts::FRAC_PI_4 * x_disk) / rest;
    let removed = r * r * rest * std::f64::consts::TAU * x_removed;
    std::f64::consts::PI * big_r * big_r * height - removed
}

/// The straight loft's round is the torus of a cylinder against a plane.
#[test]
fn a_straight_circular_loft_fillet_is_the_cylinder_round() {
    let (_secs, _rounds, _bound, _points, vol) = fillet_rim(0.0);
    let expected = straight_round_volume();
    assert!((vol - expected).abs() < 1e-4, "{vol} vs {expected}");
}

/// The tilted loft's side is not a cylinder. The round still builds, and its net stays
/// within the certificate's target (about 4e-6 mm here). The direct quotient spent
/// almost all of the fillet's time enclosing the feet; the affine enclosure builds the
/// same body without blowing the net up toward the 4096-station cap.
#[test]
fn a_tilted_circular_loft_fillet_matches_the_direct_enclosure() {
    let (_secs, rounds, bound, points, vol) = fillet_rim(2.1256);
    // Direct enclosure on this kernel: volume 490.952739 mm³, 1137 control points.
    assert!((vol - 490.952739).abs() < 1e-3, "{vol}");
    assert_eq!(rounds, 1, "one round face");
    assert!(bound <= 4.0e-6, "bound {bound}");
    assert!(points < 2500, "the round's net has {points} control points");
}
