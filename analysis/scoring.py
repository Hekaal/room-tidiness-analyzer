def compute_tidiness_score(f):
    n_obj = min(f["total_objects"] / 15, 1)
    n_area = min(f["total_bbox_area_ratio"] / 0.5, 1)
    n_floor = min(f["objects_on_floor"] / 8, 1)
    n_surface = min(f["objects_on_surface"] / 5, 1)

    # NEW
    n_floor_texture = min(f["floor_edge_density"] / 0.12, 1)
    n_floor_clutter = min(f["floor_clutter_ratio"] / 0.15, 1)

    clutter = (
        0.2 * n_obj +
        0.2 * n_area +
        0.15 * n_floor +
        0.15 * n_surface +
        0.15 * n_floor_texture +
        0.15 * n_floor_clutter
    )

    # ekstra penalti kalau inferred clutter aktif
    if f["inferred_floor_clutter"]:
        clutter += 0.1

    clutter = min(clutter, 1)
    score = int((1 - clutter) * 100)

    label = "RAPI" if score >= 71 else "SEDANG" if score >= 41 else "BERANTAKAN"

    return {
        "score": score,
        "label": label,
        "explanation": (
            f"{f['total_objects']} objek, "
            f"{f['objects_on_floor']} di lantai, "
            f"floor edge density {f['floor_edge_density']}, "
            f"floor clutter ratio {f['floor_clutter_ratio']}"
        )
    }
