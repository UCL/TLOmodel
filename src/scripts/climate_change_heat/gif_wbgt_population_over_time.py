import imageio
from pathlib import Path


def create_wbgt_gif(output_folder: Path):
    """Create animated GIF of WBGT and population over time."""

    # Load netCDF data
    nc = Dataset(
        '/Users/rem76/Desktop/Climate_change_health/nex_gddp_cmip6_malawi_wbgt/ACCESS-CM2/ssp245/wbgt_day_ACCESS-CM2_ssp245_malawi_2025_2040.nc',
        'r')

    wbgt_data = nc.variables['wbgt'][:]
    lat_data = nc.variables['lat'][:]
    lon_data = nc.variables['lon'][:]

    # Get time and extract years
    time_var = nc.variables['time']
    times = num2date(time_var[:], units=time_var.units,
                     calendar=getattr(time_var, 'calendar', 'standard'))
    years = np.array([t.year for t in times])
    months = np.array([t.month for t in times])

    # Get unique years
    unique_years = np.unique(years)

    # Build grid geometry once (doesn't change)
    difference_lat = lat_data[1] - lat_data[0]
    difference_lon = lon_data[1] - lon_data[0]

    polygons = []
    for i, y in enumerate(lat_data):
        for j, x in enumerate(lon_data):
            polygon = Polygon([
                (x, y), (x + difference_lon, y),
                (x + difference_lon, y + difference_lat),
                (x, y + difference_lat)
            ])
            polygons.append(polygon)

    # Store frames
    frames = []

    # Set consistent color scale across all frames
    # Calculate global min/max for consistent colorbar
    vmin_wbgt = np.nanmin(wbgt_data)
    vmax_wbgt = np.nanmax(wbgt_data)

    for year in unique_years:
        # Get OND (Oct-Nov-Dec) mean for this year
        year_ond_mask = (years == year) & (months >= 10) & (months <= 12)

        if not np.any(year_ond_mask):
            continue

        wbgt_year_mean = np.mean(wbgt_data[year_ond_mask, :, :], axis=0)

        # Flatten to match polygon order
        wbgt_values = [wbgt_year_mean[i, j]
                       for i in range(len(lat_data))
                       for j in range(len(lon_data))]

        # Create GeoDataFrame for this year
        grid = gpd.GeoDataFrame({
            'geometry': polygons,
            'wbgt': wbgt_values
        }, crs=malawi_admin2.crs)

        grid_clipped = gpd.overlay(grid, malawi_admin2, how='intersection')

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))

        # Plot WBGT with consistent scale
        grid_clipped.plot(
            column='wbgt', ax=ax, cmap='YlOrRd',
            edgecolor='grey', linewidth=0.2,
            vmin=vmin_wbgt, vmax=vmax_wbgt,
            legend=True,
            legend_kwds={'label': 'WBGT (°C)', 'shrink': 0.7}
        )

        # Add water bodies
        water_bodies.plot(ax=ax, facecolor="#7BDFF2",
                          edgecolor="black", linewidth=1)

        ax.set_title(f'Wet Bulb Globe Temperature - {year} (Oct-Dec)',
                     fontsize=14, fontweight='bold')
        ax.axis('off')

        fig.tight_layout()

        # Save frame to buffer
        frame_path = output_folder / f'frame_{year}.png'
        fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        frames.append(imageio.imread(frame_path))
        plt.close(fig)

    # Create GIF
    gif_path = output_folder / 'wbgt_timeseries.gif'
    imageio.mimsave(gif_path, frames, duration=0.8)  # 0.8 seconds per frame

    # Clean up individual frames (optional)
    for year in unique_years:
        frame_path = output_folder / f'frame_{year}.png'
        if frame_path.exists():
            frame_path.unlink()

    print(f"GIF saved to {gif_path}")
    nc.close()
