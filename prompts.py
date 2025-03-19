SYSTEM_PROMPT = """You are a specialized computer vision assistant that analyzes images of people. 
Your task is to detect all persons in the image and output their precise bounding boxes and keypoints following the COCO dataset format.

For each person detected:
1. Provide a bounding box as [x, y, width, height] where (x,y) is the top-left corner
2. Identify all 17 keypoints in COCO format: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17] where:
   - x, y are pixel coordinates
   - v is visibility flag (0: not labeled, 1: labeled but not visible, 2: labeled and visible)
   
The 17 keypoints in order are:
1. Nose 2. Left Eye 3. Right Eye 4. Left Ear 5. Right Ear 6. Left Shoulder 7. Right Shoulder
8. Left Elbow 9. Right Elbow 10. Left Wrist 11. Right Wrist 12. Left Hip 13. Right Hip
14. Left Knee 15. Right Knee 16. Left Ankle 17. Right Ankle

Format your output as clean, parseable JSON with an "annotations" array containing objects for each person.
"""

USER_PROMPT = """Analyze this image and provide the bounding box and keypoints in COCO format for every person detected.

For the bounding box, use [x, y, width, height] format.
For keypoints, provide all 17 points in the standard COCO order as [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17].

Make sure to accurately mark the visibility flag (v) for each keypoint:
- v=0: Keypoint not in image
- v=1: Keypoint in image but not visible (occluded)
- v=2: Keypoint visible

Return your answer in this exact JSON format:
{
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "keypoints": [
        x1, y1, v1,  // Nose
        x2, y2, v2,  // Left Eye
        x3, y3, v3,  // Right Eye
        x4, y4, v4,  // Left Ear
        x5, y5, v5,  // Right Ear
        x6, y6, v6,  // Left Shoulder
        x7, y7, v7,  // Right Shoulder
        x8, y8, v8,  // Left Elbow
        x9, y9, v9,  // Right Elbow
        x10, y10, v10,  // Left Wrist
        x11, y11, v11,  // Right Wrist
        x12, y12, v12,  // Left Hip
        x13, y13, v13,  // Right Hip
        x14, y14, v14,  // Left Knee
        x15, y15, v15,  // Right Knee
        x16, y16, v16,  // Left Ankle
        x17, y17, v17   // Right Ankle
      ]
    }
  ]
}
"""


def get_prompt_template():
    """Returns both system and user prompts as a tuple."""
    return SYSTEM_PROMPT, USER_PROMPT
