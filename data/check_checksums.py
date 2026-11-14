import hashlib
import os
import json


def calculate_checksum(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    data_dir = os.environ["ELLIPTIC_DATA_DIR"]

    reference_path = os.path.join(
        os.path.dirname(__file__), "checksums.json"
    )

    with open(reference_path, "r") as file:
        expected_checksums = json.load(file)

    all_valid = True

    for filename, expected in expected_checksums.items():
        file_path = os.path.join(data_dir, filename)
        actual = calculate_checksum(file_path)

        if actual == expected:
            print(filename, "PASSED")
        else:
            print(filename, "FAILED")
            all_valid = False

    if all_valid:
        print("CHECKSUM VALIDATION PASSED")
    else:
        raise ValueError("Checksum validation failed")


if __name__ == "__main__":
    main()