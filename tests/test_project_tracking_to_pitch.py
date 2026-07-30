from scripts.project_tracking_to_pitch import calibration_image_scale


def test_calibration_image_scale_normalizes_detection_coordinates():
    camera = {"principal_point": [960.0, 540.0]}

    assert calibration_image_scale(camera, 1280, 720) == (1.5, 1.5)


def test_calibration_image_scale_is_identity_at_calibration_resolution():
    camera = {"principal_point": [960.0, 540.0]}

    assert calibration_image_scale(camera, 1920, 1080) == (1.0, 1.0)
