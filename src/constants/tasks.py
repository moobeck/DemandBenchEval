from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from src.configurations.utils.enums import DatasetName, FrequencyType, HierarchyType
from src.constants.cross_validation_windows import TrainTestSplitConfig, DEFAULT_TRAIN_TEST_SPLIT
from src.dataset.dataset_loader import DatasetLoader
from functools import cached_property

class TaskName(Enum):
    M5_PRODUCT_WEEKLY_4 = "m5_product_weekly_4"
    M5_PRODUCT_MONTHLY_3 = "m5_product_monthly_3"
    M5_STORE_DAILY_7 = "m5_store_daily_7"
    FAVORITA_PRODUCT_WEEKLY_4 = "favorita_product_weekly_4"
    FAVORITA_PRODUCT_MONTHLY_3 = "favorita_product_monthly_3"
    FAVORITA_STORE_DAILY_7 = "favorita_store_daily_7"
    FAVORITA_STORE_WEEKLY_4 = "favorita_store_weekly_4"
    ROHLIK_PRODUCT_STORE_WEEKLY_4 = "rohlik_product_store_weekly_4"
    ROHLIK_PRODUCT_DAILY_7 = "rohlik_product_daily_7"
    ROHLIK_PRODUCT_WEEKLY_4 = "rohlik_product_weekly_4"
    ROSSMANN_PRODUCT_STORE_WEEKLY_4 = "rossmann_product_store_weekly_4"
    ROSSMANN_STORE_WEEKLY_4 = "rossmann_store_weekly_4"
    BAKERY_PRODUCT_STORE_DAILY_7 = "bakery_product_store_daily_7"
    BAKERY_PRODUCT_STORE_WEEKLY_4 = "bakery_product_store_weekly_4"
    BAKERY_PRODUCT_DAILY_7 = "bakery_product_daily_7"
    BAKERY_STORE_DAILY_7 = "bakery_store_daily_7"
    BAKERY_STORE_WEEKLY_4 = "bakery_store_weekly_4"
    YAZ_PRODUCT_DAILY_7 = "yaz_product_daily_7"
    PHARMACY_PRODUCT_WEEKLY_4 = "pharmacy_product_weekly_4"
    PHARMACY2_PRODUCT_STORE_DAILY_7 = "pharmacy2_product_store_daily_7"
    PHARMACY2_PRODUCT_STORE_WEEKLY_4 = "pharmacy2_product_store_weekly_4"
    FRESHRETAIL50K_PRODUCT_DAILY_7 = "freshretail50k_product_daily_7"
    FRESHRETAIL50K_STORE_DAILY_7 = "freshretail50k_store_daily_7"
    HOTEL_PRODUCT_STORE_DAILY_7 = "hotel_product_store_daily_7"
    HOTEL_PRODUCT_STORE_WEEKLY_4 = "hotel_product_store_weekly_4"
    HOTEL_PRODUCT_DAILY_7 = "hotel_product_daily_7"
    HOTEL_STORE_DAILY_7 = "hotel_store_daily_7"
    HOTEL_STORE_WEEKLY_4 = "hotel_store_weekly_4"
    ONLINERETAIL_PRODUCT_WEEKLY_4 = "onlineretail_product_weekly_4"
    ONLINERETAIL2_PRODUCT_WEEKLY_4 = "onlineretail2_product_weekly_4"
    AUSTRALIANRETAIL_PRODUCT_STORE_MONTHLY_3 = "australianretail_product_store_monthly_3"
    AUSTRALIANRETAIL_PRODUCT_MONTHLY_3 = "australianretail_product_monthly_3"
    AUSTRALIANRETAIL_STORE_MONTHLY_3 = "australianretail_store_monthly_3"
    KAGGLEDEMAND_PRODUCT_STORE_WEEKLY_4 = "kaggledemand_product_store_weekly_4"
    KAGGLEDEMAND_STORE_WEEKLY_4 = "kaggledemand_store_weekly_4"
    PRODUCTDEMAND_PRODUCT_STORE_WEEKLY_4 = "productdemand_product_store_weekly_4"
    PRODUCTDEMAND_PRODUCT_STORE_MONTHLY_3 = "productdemand_product_store_monthly_3"
    PRODUCTDEMAND_PRODUCT_WEEKLY_4 = "productdemand_product_weekly_4"
    PRODUCTDEMAND_PRODUCT_MONTHLY_3 = "productdemand_product_monthly_3"
    VN1_PRODUCT_WEEKLY_4 = "vn1_product_weekly_4"
    KAGGLERETAIL_PRODUCT_STORE_WEEKLY_4 = "kaggleretail_product_store_weekly_4"
    KAGGLERETAIL_PRODUCT_WEEKLY_4 = "kaggleretail_product_weekly_4"
    KAGGLERETAIL_STORE_WEEKLY_4 = "kaggleretail_store_weekly_4"
    KAGGLEWALMART_STORE_WEEKLY_4 = "kagglewalmart_store_weekly_4"
    HIERARCHICALSALES_PRODUCT_DAILY_7 = "hierarchicalsales_product_daily_7"
    HIERARCHICALSALES_PRODUCT_WEEKLY_4 = "hierarchicalsales_product_weekly_4"
    HIERARCHICALSALES_PRODUCT_MONTHLY_3 = "hierarchicalsales_product_monthly_3"
    CARPARTS_PRODUCT_MONTHLY_3 = "carparts_product_monthly_3"
    FOSSIL_PRODUCT_MONTHLY_3 = "fossil_product_monthly_3"


@dataclass(frozen=True)
class Task:
    name: TaskName
    dataset_name: DatasetName
    hierarchy: HierarchyType
    frequency: FrequencyType
    forecast_horizon: int
    train_test_split: TrainTestSplitConfig = field(
        default_factory=lambda: DEFAULT_TRAIN_TEST_SPLIT
    )

    @cached_property
    def dataset(self) -> DatasetName:

        dataset = DatasetLoader.load(self.dataset_name)

        dataset = dataset.aggregate_frequency(
            FrequencyType.get_alias(
                self.frequency, context="demandbench"
            )
        )

        dataset = dataset.aggregate_hierarchy(
            HierarchyType.get_alias(
                self.hierarchy, context="demandbench"
            )
        ) if self.hierarchy != HierarchyType.PRODUCT_STORE else dataset

        return dataset


# parser class with one method per argument
class TaskNameParser:
    def __init__(self, value: str):
        self.value = value
        self.parts = value.split("_")
        if len(self.parts) < 4:
            raise ValueError(f"unexpected task name format: {value}")

    def parse_dataset(self) -> DatasetName:
        return DatasetName[self.parts[0].upper()]

    def parse_hierarchy(self) -> HierarchyType:
        hierarchy_token = "_".join(self.parts[1:-2]).upper()
        return HierarchyType[hierarchy_token]

    def parse_frequency(self) -> FrequencyType:
        return FrequencyType[self.parts[-2].upper()]

    def parse_horizon(self) -> int:
        return int(self.parts[-1])

    def parse_all(self):
        return (
            
            self.parse_dataset(),
            self.parse_hierarchy(),
            self.parse_frequency(),
            self.parse_horizon(),
        )


TASKS: dict[str, Task] = {
    task_enum.value: Task(
        name=task_enum,
        dataset_name=TaskNameParser(task_enum.value).parse_dataset(),
        hierarchy=TaskNameParser(task_enum.value).parse_hierarchy(),
        frequency=TaskNameParser(task_enum.value).parse_frequency(),
        forecast_horizon=TaskNameParser(task_enum.value).parse_horizon(),
    )
    for task_enum in TaskName
}



__all__ = ["Task", "TaskName", "TASKS"]
