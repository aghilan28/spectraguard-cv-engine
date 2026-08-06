import os
import json

def main():
    root_dir = "storage/events"
    if not os.path.exists(root_dir):
        print("No events directory found.")
        return
        
    removed_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    if os.path.getsize(path) > 2:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("camera_name") == "test_cam":
                                os.remove(path)
                                removed_count += 1
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    print(f"Removed {removed_count} test event files.")

if __name__ == "__main__":
    main()
