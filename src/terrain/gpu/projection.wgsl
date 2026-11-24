// Exact spherical geometry in a local west/north/up tangent basis. The
// half-angle identity avoids subtracting two nearly equal planet radii.
struct GridProjection {
    sphere:vec4<f32>,
    grid:vec4<f32>,
}
fn grid_position(p:GridProjection, xy:vec2<u32>, bed:f32)->vec3<f32>{
    let offset=vec2<f32>(xy)*p.grid.x-vec2<f32>(p.grid.y);
    let r=p.sphere.x;
    let dlat=offset.y/r;
    let dlon=offset.x/(r*p.sphere.w);
    let sinlat=sin(dlat);
    let coslat=p.sphere.z*cos(dlat)-p.sphere.y*sinlat;
    let sin_half_lat=sin(dlat*0.5);
    let sin_half_lon=sin(dlon*0.5);
    let sag_lon=2.0*sin_half_lon*sin_half_lon;
    let sag_lat=2.0*sin_half_lat*sin_half_lat;
    let surface_r=r+bed;
    return vec3<f32>(
        p.grid.y+surface_r*coslat*sin(dlon),
        p.grid.y+surface_r*(sinlat+p.sphere.y*coslat*sag_lon),
        bed-surface_r*(sag_lat+p.sphere.z*coslat*sag_lon)
    );
}
