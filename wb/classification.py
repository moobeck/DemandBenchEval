import json
from demandbench.datasets import (
    load_m5,
    load_favorita,
    load_rohlik,
    load_rossmann,
    load_bakery,
    load_yaz,
    load_pharmacy,
    load_freshretail50k,
    load_hoteldemand,
    load_onlineretail,
    load_onlineretail2,
    load_hierarchicalsales,
    load_australianretail,
    load_carparts,
    load_kaggledemand,
    load_productdemand,
    load_pharmacy2,
    load_vn1,
    load_kagglewalmart,
    load_fossil,
)

dataset_loaders = {
    "m5": load_m5,
    "favorita": load_favorita,
    "rohlik": load_rohlik,
    "rossmann": load_rossmann,
    "bakery": load_bakery,
    "yaz": load_yaz,
    "pharmacy": load_pharmacy,
    "freshretail50k": load_freshretail50k,
    "hoteldemand": load_hoteldemand,
    "onlineretail": load_onlineretail,
    "onlineretail2": load_onlineretail2,
    "hierarchicalsales": load_hierarchicalsales,
    "australianretail": load_australianretail,
    "carparts": load_carparts,
    "kaggledemand": load_kaggledemand,
    "productdemand": load_productdemand,
    "pharmacy2": load_pharmacy2,
    "vn1": load_vn1,
    "kagglewalmart": load_kagglewalmart,
    "fossil": load_fossil,
}

dataset_to_hierarchy = {
    "m5": "product",
    "favorita": "store",
    "rohlik": "product",
    "rossmann": "product_store",
    "bakery": "store",
    "yaz": "product",
    "pharmacy": "product",
    "freshretail50k": "product",
    "hoteldemand": "product_store",
    "onlineretail": "product",
    "onlineretail2": "product",
    "hierarchicalsales": "product",
    "australianretail": "product_store",
    "carparts": "product",
    "productdemand": "product_store",
    "pharmacy2": "product_store",
    "vn1": "product",
    "kagglewalmart": "store",
    "kaggledemand": "product_store",
    "fossil": "product",
}

dataset_to_frequency = {
    "m5": "monthly",
    "favorita": "weekly",
    "rohlik": "daily",
    "rossmann": "weekly",
    "bakery": "weekly",
    "yaz": "daily",
    "pharmacy": "weekly",
    "freshretail50k": "daily",
    "hoteldemand": "daily",
    "onlineretail": "weekly",
    "onlineretail2": "weekly",
    "hierarchicalsales": "daily",
    "australianretail": "monthly",
    "carparts": "monthly",
    "productdemand": "monthly",
    "pharmacy2": "weekly",
    "vn1": "weekly",
    "kagglewalmart": "weekly",
    "kaggledemand": "weekly",
    "fossil": "monthly",
}


CLASSIFICATION_FILEPATH = "artifacts/classification/"

def get_classification_by_id(dataset_name: str):
    loader = dataset_loaders.get(dataset_name)
    if loader is None:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    data = loader()

    hierarchy = dataset_to_hierarchy[dataset_name]
    frequency = dataset_to_frequency[dataset_name]

    data = data.aggregate_hierarchy(hierarchy) if hierarchy != "product_store" else data
    data = data.aggregate_frequency(frequency)

    metadata = data.metadata
    intermittent_ids = list(metadata.intermittent_ids)
    lumpy_ids = list(metadata.lumpy_ids)
    smooth_ids = list(metadata.smooth_ids)
    erratic_ids = list(metadata.erratic_ids)

    # Combine intermittent and lumpy ids
    intermittent_and_lumpy_ids = list(set(intermittent_ids) | set(lumpy_ids))
    return intermittent_ids, lumpy_ids, smooth_ids, erratic_ids, intermittent_and_lumpy_ids
    



if __name__ == "__main__":
    intermittent_by_dataset = {}
    lumpy_by_dataset = {}
    smooth_by_dataset = {}
    erratic_by_dataset = {}

    lumpy_and_intermittent_by_dataset = {}

    for name in dataset_loaders.keys():
        intermittent_ids, lumpy_ids, smooth_ids, erratic_ids, intermittent_and_lumpy_ids = get_classification_by_id(name)
        intermittent_by_dataset[name] = intermittent_ids
        lumpy_by_dataset[name] = lumpy_ids
        smooth_by_dataset[name] = smooth_ids
        erratic_by_dataset[name] = erratic_ids
        lumpy_and_intermittent_by_dataset[name] = intermittent_and_lumpy_ids
        


    with open(f"{CLASSIFICATION_FILEPATH}intermittent_ids.json", "w") as f:
        json.dump(intermittent_by_dataset, f, indent=4)

    with open(f"{CLASSIFICATION_FILEPATH}lumpy_ids.json", "w") as f:
        json.dump(lumpy_by_dataset, f, indent=4)

    with open(f"{CLASSIFICATION_FILEPATH}smooth_ids.json", "w") as f:
        json.dump(smooth_by_dataset, f, indent=4)

    with open(f"{CLASSIFICATION_FILEPATH}erratic_ids.json", "w") as f:
        json.dump(erratic_by_dataset, f, indent=4)

    with open(f"{CLASSIFICATION_FILEPATH}lumpy_and_intermittent_ids.json", "w") as f:
        json.dump(lumpy_and_intermittent_by_dataset, f, indent=4)
