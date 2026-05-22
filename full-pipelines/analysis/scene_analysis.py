"""
Scene Analysis Generator
========================
Consolidate dialogue, complexity, and obstruction data into SceneAnalysis.json.
"""

import json

from config import logger


def generate_scene_analysis(scenes_data, dialogue_set, complexity, obstruction, output_path):
    """Generate the consolidated SceneAnalysis.json."""
    analysis = []
    for scene in scenes_data["scenes"]:
        sn = scene["scene"]
        have_audio = sn in dialogue_set
        comp = complexity.get(sn, {})
        obs = obstruction.get(sn, {})

        avg_face = comp.get("avg_face_count", 0)
        person_speaking = have_audio and avg_face > 0
        effective_max = comp.get("effective_max_persons", comp.get("max_face_count", 0))
        multiple_person = effective_max > 1

        analysis.append({
            "scene": sn,
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration_seconds": scene["duration_seconds"],
            "haveAudio": have_audio,
            "personSpeaking": person_speaking,
            "multiplePersonInScene": multiple_person,
            "haveObstructionInLip": obs.get("is_obstructed", False),
            "isComplexScene": comp.get("is_complex", False),
        })

    report = {
        "video": scenes_data["video"],
        "total_scenes": len(analysis),
        "summary": {
            "scenes_with_audio": sum(1 for s in analysis if s["haveAudio"]),
            "person_speaking": sum(1 for s in analysis if s["personSpeaking"]),
            "multiple_persons": sum(1 for s in analysis if s["multiplePersonInScene"]),
            "lip_obstruction": sum(1 for s in analysis if s["haveObstructionInLip"]),
            "complex_scenes": sum(1 for s in analysis if s["isComplexScene"]),
        },
        "scenes": analysis,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"SceneAnalysis.json saved: {output_path}")
    logger.info(f"  Audio: {report['summary']['scenes_with_audio']}, "
                f"Speaking: {report['summary']['person_speaking']}, "
                f"Multi: {report['summary']['multiple_persons']}, "
                f"Obstruction: {report['summary']['lip_obstruction']}, "
                f"Complex: {report['summary']['complex_scenes']}")
    return analysis
