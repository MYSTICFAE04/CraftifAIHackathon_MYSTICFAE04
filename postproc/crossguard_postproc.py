#!/usr/bin/env python3
"""
CrossGuard Edge: Conflict Zone Post-Processor
Analyzes DeepStream detection and tracking metadata against a defined crosswalk conflict zone.
Detects simultaneous pedestrian and vehicle occupancy, applies temporal smoothing (sustained frame threshold),
and emits deduplicated safety event alerts.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm to test if (x, y) is inside a polygon."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_anchor_point(bbox: Dict[str, float], anchor_type: str = "bottom_center") -> Tuple[float, float]:
    """Calculate the representative ground-plane anchor point for a 2D bounding box."""
    left = bbox["left"]
    top = bbox["top"]
    width = bbox["width"]
    height = bbox["height"]

    if anchor_type == "bottom_center":
        return left + (width / 2.0), top + height
    elif anchor_type == "center":
        return left + (width / 2.0), top + (height / 2.0)
    elif anchor_type == "bottom_left":
        return left, top + height
    else:
        return left + (width / 2.0), top + height


def process_safety_events(
    metadata_path: str,
    config_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """Load metadata and config, detect conflict events, and write out safety events."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    camera_id = config.get("camera_id", "cam_default")
    polygon = config.get("conflict_zone_polygon", [])
    ped_labels = set(config.get("pedestrian_labels", ["person", "pedestrian"]))
    veh_labels = set(config.get("vehicle_labels", ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"]))
    min_confidence = config.get("min_confidence", 0.40)
    sustained_threshold = config.get("sustained_frames_threshold", 3)
    cooldown_seconds = config.get("cooldown_seconds", 3.0)
    anchor_type = config.get("bbox_anchor", "bottom_center")

    frames = meta_data.get("frames", [])
    total_frames = len(frames)

    consecutive_conflict_frames = 0
    active_conflict_peds: Dict[int, Dict[str, Any]] = {}
    active_conflict_vehs: Dict[int, Dict[str, Any]] = {}
    conflict_start_frame = None
    conflict_start_pts_ns = None

    last_alert_pts_ns = None
    cooldown_ns = int(cooldown_seconds * 1e9)

    events: List[Dict[str, Any]] = []

    for frame_idx, frame in enumerate(frames):
        pts_ns = frame.get("pts_ns", frame_idx * int(1e9 / 60))
        objects = frame.get("objects", [])

        peds_in_zone: List[Dict[str, Any]] = []
        vehs_in_zone: List[Dict[str, Any]] = []

        for obj in objects:
            conf = obj.get("confidence", 0.0)
            if conf < min_confidence:
                continue

            label = obj.get("label", "").lower()
            bbox = obj.get("bbox", {})
            if not bbox:
                continue

            ax, ay = get_anchor_point(bbox, anchor_type)
            if point_in_polygon(ax, ay, polygon):
                obj_summary = {
                    "track_id": obj.get("id"),
                    "label": label,
                    "confidence": round(conf, 4),
                    "anchor": [round(ax, 1), round(ay, 1)],
                    "bbox": bbox,
                }
                if label in ped_labels:
                    peds_in_zone.append(obj_summary)
                elif label in veh_labels:
                    vehs_in_zone.append(obj_summary)

        is_conflict = len(peds_in_zone) > 0 and len(vehs_in_zone) > 0

        if is_conflict:
            consecutive_conflict_frames += 1
            if conflict_start_frame is None:
                conflict_start_frame = frame.get("frame", frame_idx)
                conflict_start_pts_ns = pts_ns

            for p in peds_in_zone:
                tid = p["track_id"]
                if tid is not None:
                    active_conflict_peds[tid] = p
            for v in vehs_in_zone:
                tid = v["track_id"]
                if tid is not None:
                    active_conflict_vehs[tid] = v

            # Check if threshold reached and not in cooldown
            if consecutive_conflict_frames == sustained_threshold:
                can_trigger = True
                if last_alert_pts_ns is not None:
                    if pts_ns - last_alert_pts_ns < cooldown_ns:
                        can_trigger = False

                if can_trigger:
                    last_alert_pts_ns = pts_ns
                    event = {
                        "event_id": f"EVT_{len(events) + 1:04d}",
                        "camera_id": camera_id,
                        "type": "PEDESTRIAN_VEHICLE_CONFLICT",
                        "severity": "CRITICAL",
                        "trigger_frame": frame.get("frame", frame_idx),
                        "start_frame": conflict_start_frame,
                        "timestamp_pts_ns": pts_ns,
                        "timestamp_sec": round(pts_ns / 1e9, 3) if pts_ns else round(frame_idx / 60.0, 3),
                        "conflict_zone": "crosswalk_conflict_zone",
                        "pedestrians_in_zone_count": len(peds_in_zone),
                        "vehicles_in_zone_count": len(vehs_in_zone),
                        "pedestrian_tracks": list(active_conflict_peds.keys()),
                        "vehicle_tracks": list(active_conflict_vehs.keys()),
                        "sample_pedestrian": peds_in_zone[0],
                        "sample_vehicle": vehs_in_zone[0],
                    }
                    events.append(event)
        else:
            consecutive_conflict_frames = 0
            active_conflict_peds.clear()
            active_conflict_vehs.clear()
            conflict_start_frame = None
            conflict_start_pts_ns = None

    result = {
        "summary": {
            "camera_id": camera_id,
            "total_frames_analyzed": total_frames,
            "sustained_threshold_frames": sustained_threshold,
            "cooldown_seconds": cooldown_seconds,
            "total_safety_events_triggered": len(events),
        },
        "events": events,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossGuard Edge Safety Post-Processor")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_meta = os.path.join(base_dir, "pipeline", "deepstream", "output", "metadata.json")
    default_cfg = os.path.join(base_dir, "postproc", "zone_config.json")
    default_out = os.path.join(base_dir, "postproc", "safety_events.json")

    parser.add_argument("--metadata", default=default_meta, help="Path to DeepStream metadata.json")
    parser.add_argument("--config", default=default_cfg, help="Path to zone_config.json")
    parser.add_argument("--output", default=default_out, help="Path to output safety_events.json")

    args = parser.parse_args()

    print(f"[CrossGuard PostProc] Ingesting metadata: {args.metadata}")
    print(f"[CrossGuard PostProc] Ingesting config:   {args.config}")
    res = process_safety_events(args.metadata, args.config, args.output)
    print(f"[CrossGuard PostProc] Analysis complete!")
    print(f"[CrossGuard PostProc] Frames analyzed: {res['summary']['total_frames_analyzed']}")
    print(f"[CrossGuard PostProc] Safety events triggered: {res['summary']['total_safety_events_triggered']}")
    print(f"[CrossGuard PostProc] Output saved to: {args.output}")


if __name__ == "__main__":
    main()
