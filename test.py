# ==============================================================================
#                 Pytest Tests for Part Identifier
# ==============================================================================
#
# Description:
# This file contains pytest tests for the core functions of the terminal-based
# part identifier. It uses fixtures to create a controlled test environment
# and parameterized tests to check the identification logic.
#
# How to Run:
# 1. Save the main script as `terminal_part_identifier.py`.
# 2. Save this test script as `test_part_identifier.py` in the same directory.
# 3. Install pytest: `pip install pytest`
# 4. Run the tests from your terminal: `pytest`
#
# ==============================================================================

import os
import numpy as np
import pytest
from PIL import Image, ImageDraw

# Import the functions to be tested from the main script
# Make sure the main script is saved as 'terminal_part_identifier.py'
from main import (
    get_feature_extractor,
    extract_features,
    index_inventory,
    identify_part,
    INVENTORY_DIR,
    FEATURES_FILE,
    NAMES_FILE
)

@pytest.fixture(scope="module")
def feature_extractor():
    """
    Pytest fixture to load the feature extraction model once for all tests.
    This is efficient as model loading is slow.
    """
    return get_feature_extractor()


@pytest.fixture
def setup_test_environment(tmp_path):
    """
    Pytest fixture to create a temporary, isolated test environment.
    This runs for each test function to ensure no side effects.

    ARRANGE
    """
    # Use a temporary directory provided by pytest's tmp_path fixture
    test_inventory_dir = tmp_path / "test_inventory"
    os.makedirs(test_inventory_dir)

    # Monkeypatch the global variables to use the temporary paths
    original_inventory_dir = INVENTORY_DIR
    original_features_file = FEATURES_FILE
    original_names_file = NAMES_FILE

    # Point the script's constants to our temporary paths
    globals()['INVENTORY_DIR'] = str(test_inventory_dir)
    globals()['FEATURES_FILE'] = str(tmp_path / "test_features.npy")
    globals()['NAMES_FILE'] = str(tmp_path / "test_names.npy")

    # Create dummy part directories and images
    part_a_dir = test_inventory_dir / "test_part_A"
    part_b_dir = test_inventory_dir / "test_part_B"
    os.makedirs(part_a_dir)
    os.makedirs(part_b_dir)

    # Create an image for Part A
    img_a = Image.new('RGB', (100, 100), color='red')
    img_a_path = part_a_dir / "sample_a.png"
    img_a.save(img_a_path)

    # Create an image for Part B
    img_b = Image.new('RGB', (100, 100), color='blue')
    img_b_path = part_b_dir / "sample_b.png"
    img_b.save(img_b_path)

    # Yield the paths for the test to use
    yield {
        "inventory_dir": str(test_inventory_dir),
        "part_a_image": str(img_a_path),
        "part_b_image": str(img_b_path),
    }

    # Teardown: Restore original global paths after the test
    globals()['INVENTORY_DIR'] = original_inventory_dir
    globals()['FEATURES_FILE'] = original_features_file
    globals()['NAMES_FILE'] = original_names_file


def test_extract_features(feature_extractor, setup_test_environment):
    """
    Tests that the extract_features function returns a vector of the correct shape.
    """
    # ARRANGE
    image_path = setup_test_environment["part_a_image"]
    expected_feature_length = 2048  # ResNet50 with avg pooling outputs this length

    # ACT
    features = extract_features(image_path, feature_extractor)

    # ASSERT
    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert features.ndim == 1, "Feature vector should be 1-dimensional"
    assert len(features) == expected_feature_length, f"Feature vector should have length {expected_feature_length}"


def test_index_inventory(feature_extractor, setup_test_environment, capsys):
    """
    Tests that the index_inventory function creates the feature and name files.
    """
    # ARRANGE
    # The setup_test_environment fixture arranges the directory and files.

    # ACT
    index_inventory(feature_extractor)

    # ASSERT
    # Check that the output files were created in the temporary directory
    assert os.path.exists(globals()['FEATURES_FILE']), "Features file was not created"
    assert os.path.exists(globals()['NAMES_FILE']), "Names file was not created"

    features = np.load(globals()['FEATURES_FILE'])
    names = np.load(globals()['NAMES_FILE'])

    assert len(features) == 2, "Should have indexed 2 features"
    assert len(names) == 2, "Should have indexed 2 names"
    assert "test_part_A" in names
    assert "test_part_B" in names


@pytest.mark.parametrize("image_to_test, expected_name", [
    ("part_a_image", "test_part_A"),
    ("part_b_image", "test_part_B"),
])
def test_identify_part(feature_extractor, setup_test_environment, capsys, image_to_test, expected_name):
    """
    Tests the end-to-end identification logic for known parts.
    """
    # ARRANGE
    # Index the temporary inventory
    index_inventory(feature_extractor)

    # Get the path of the image to be identified for this test case
    image_path = setup_test_environment[image_to_test]

    # ACT
    # Run the identification function
    identify_part(image_path, feature_extractor)

    # ASSERT
    # Capture the printed output to verify the result
    captured = capsys.readouterr()
    output = captured.out

    assert f"Identified as:   {expected_name}" in output, f"Should have identified the part as {expected_name}"

    # Extract confidence and check if it's high
    confidence_line = [line for line in output.split('\n') if "Confidence" in line][0]
    confidence_str = confidence_line.split(':')[1].strip().replace('%', '')
    confidence = float(confidence_str)

    assert confidence > 95.0, "Confidence for a known part should be very high (> 95%)"