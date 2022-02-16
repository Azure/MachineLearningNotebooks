import json
import os
import sys
import argparse

# Define Converters


class CocoToJSONLinesConverter:
    def convert(self):
        raise NotImplementedError


class BoundingBoxConverter(CocoToJSONLinesConverter):
    def __init__(self, coco_data):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        self.image_id_to_data_index = {}
        for i in range(0, len(coco_data["images"])):
            self.json_lines_data.append({})
            self.json_lines_data[i]["image_url"] = ""
            self.json_lines_data[i]["image_details"] = {}
            self.json_lines_data[i]["label"] = []
        for i in range(0, len(coco_data["categories"])):
            self.categories[coco_data["categories"][i]["id"]] = coco_data["categories"][
                i
            ]["name"]

    def _populate_image_url(self, index, coco_image):
        self.json_lines_data[index]["image_url"] = coco_image["file_name"]
        self.image_id_to_data_index[coco_image["id"]] = index

    def _populate_image_details(self, index, coco_image):
        file_name = coco_image["file_name"]
        self.json_lines_data[index]["image_details"]["format"] = file_name[
            file_name.rfind(".") + 1 :
        ]
        self.json_lines_data[index]["image_details"]["width"] = coco_image["width"]
        self.json_lines_data[index]["image_details"]["height"] = coco_image["height"]

    def _populate_bbox_in_label(self, label, annotation, image_details):
        # if bbox comes as normalized, skip normalization.
        if max(annotation["bbox"]) < 1.5:
            width = 1
            height = 1
        else:
            width = image_details["width"]
            height = image_details["height"]
        label["topX"] = annotation["bbox"][0] / width
        label["topY"] = annotation["bbox"][1] / height
        label["bottomX"] = (annotation["bbox"][0] + annotation["bbox"][2]) / width
        label["bottomY"] = (annotation["bbox"][1] + annotation["bbox"][3]) / height

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation["image_id"]]
        image_details = self.json_lines_data[index]["image_details"]
        label = {"label": self.categories[annotation["category_id"]]}
        self._populate_bbox_in_label(label, annotation, image_details)
        self._populate_isCrowd(label, annotation)
        self.json_lines_data[index]["label"].append(label)

    def _populate_isCrowd(self, label, annotation):
        if "iscrowd" in annotation.keys():
            label["isCrowd"] = annotation["iscrowd"]

    def convert(self):
        for i in range(0, len(self.coco_data["images"])):
            self._populate_image_url(i, self.coco_data["images"][i])
            self._populate_image_details(i, self.coco_data["images"][i])
        for i in range(0, len(self.coco_data["annotations"])):
            self._populate_label(self.coco_data["annotations"][i])
        return self.json_lines_data


if __name__ == "__main__":
    # Parse arguments that are passed into the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_coco_file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["ObjectDetection"],
        default="ObjectDetection",
    )
    parser.add_argument("--base_url", type=str, default=None)

    args = parser.parse_args()

    input_coco_file_path = args.input_coco_file_path
    output_dir = args.output_dir
    output_file_path = output_dir + "/" + args.output_file_name
    task_type = args.task_type
    base_url = args.base_url

    def read_coco_file(coco_file):
        with open(coco_file) as f_in:
            return json.load(f_in)

    def write_json_lines(converter, filename, base_url=None):
        json_lines_data = converter.convert()
        with open(filename, "w") as outfile:
            for json_line in json_lines_data:
                if base_url is not None:
                    image_url = json_line["image_url"]
                    json_line["image_url"] = (
                        base_url + image_url[image_url.rfind("/") + 1 :]
                    )
                json.dump(json_line, outfile, separators=(",", ":"))
                outfile.write("\n")
            print(f"Conversion completed. Converted {len(json_lines_data)} lines.")

    coco_data = read_coco_file(input_coco_file_path)

    print("Converting for {}".format(task_type))

    # Defined in azureml.contrib.dataset.labeled_dataset.LabeledDatasetTask.OBJECT_DETECTION.value
    if task_type == "ObjectDetection":
        converter = BoundingBoxConverter(coco_data)
        write_json_lines(converter, output_file_path, base_url)

    else:
        print("ERROR: Invalid Task Type")
        pass
