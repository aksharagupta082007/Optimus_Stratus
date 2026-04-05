"""
Translates classifier output into a structured result
that observation_builder.py can consume.
"""

CONFIDENCE_THRESHOLD = 0.70   # below this → treat as uncertain

def postprocess(prediction: dict) -> dict:
    """
    Input:  {'label': str, 'confidence': float, 'raw_score': float}
    Output: structured result for the RL observation
    """
    label      = prediction['label']
    confidence = prediction['confidence']
    raw_score  = prediction['raw_score']

    is_cloudy    = label == 'cloudy'
    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    # Actionable flags for the RL agent
    worth_downlinking = (not is_cloudy) and (not is_uncertain)

    # CHANGE in postprocess() return dict — ADD usefulness score:
    return {
        'is_cloudy':                    is_cloudy,
        'is_uncertain':                 is_uncertain,
        'classifier_confidence':        round(confidence, 4),
        'current_frame_cloud_prob':     round(raw_score, 4),
        'current_frame_usefulness':     0.0 if is_cloudy else round(confidence, 4),
        'classifier_success':           not is_uncertain,
        'worth_downlinking':            not is_cloudy and not is_uncertain,  # keep for TransitionContext
    }

def to_transition_fields(postprocess_result: dict, ground_truth_is_cloudy: bool) -> dict:
    """
    Feeds classifier outcome into TransitionContext fields that RewardEngine reads.
    ground_truth_is_cloudy comes from simulation/cloud_model.py in the env step.
    """
    return {
        'classifier_ran':     True,
        'classifier_correct': postprocess_result['is_cloudy'] == ground_truth_is_cloudy,
    }