def compute_tidiness_score(f):
    norm_obj = min(f["total_objects"] / 15, 1)
    norm_area = min(f["total_bbox_area_ratio"] / 0.5, 1)
    norm_floor = min(f["objects_on_floor"] / 8, 1)
    norm_surface = min(f["objects_on_surface"] / 5, 1)
    norm_edge = min(f["edge_density"] / 0.15, 1)

    clutter = (
        0.25 * norm_obj +
        0.25 * norm_area +
        0.2 * norm_floor +
        0.2 * norm_surface +
        0.1 * norm_edge
    )

    score = int((1 - clutter) * 100)

    if score >= 71:
        label = "RAPI"
    elif score >= 31:
        label = "SEDANG"
    else:
        label = "BERANTAKAN"

    explanation = (
        f"{f['total_objects']} objek, "
        f"{f['objects_on_floor']} di lantai, "
        f"{f['objects_on_surface']} di kasur/sofa, "
        f"edge density {f['edge_density']}"
    )

    return {"score": score, "label": label, "explanation": explanation}
