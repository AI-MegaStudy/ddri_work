from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SHARED_DIR = ROOT_DIR / "3조 공유폴더"
OUTPUT_DIR = BASE_DIR / "outputs" / "admin_dong"

STATION_PATH = SHARED_DIR / "강남구 대여소 정보 (2023~2025)" / "2023_강남구_대여소.csv"
MAPPING_PATH = SHARED_DIR / "서울특별시행정동별 서울생활인구(내국인) 2025년" / "서울시_행정동코드_매핑표.csv"


def read_csv_with_fallback(path: Path, encodings: list[str] | None = None, **kwargs) -> pd.DataFrame:
    fallback_encodings = encodings or ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    last_error: Exception | None = None
    for encoding in fallback_encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def normalize_station_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str).replace("<NA>", pd.NA)


def build_keyword_table(mapping: pd.DataFrame) -> pd.DataFrame:
    dong = mapping.copy()
    dong["base_name"] = (
        dong["행정동명"]
        .str.replace(r"본동$", "", regex=True)
        .str.replace(r"[0-9]+동$", "동", regex=True)
    )
    return dong


def extract_text_candidates(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def assign_candidates(row: pd.Series, keyword_table: pd.DataFrame) -> pd.Series:
    text = f"{row['대여소명']} {row['주소']}"
    exact_hits = extract_text_candidates(text, keyword_table["행정동명"].drop_duplicates().tolist())
    base_hits = extract_text_candidates(text, keyword_table["base_name"].drop_duplicates().tolist())

    exact_candidates = keyword_table[keyword_table["행정동명"].isin(exact_hits)]
    base_candidates = keyword_table[keyword_table["base_name"].isin(base_hits)]

    if not exact_candidates.empty:
        candidates = exact_candidates.drop_duplicates(subset=["행정동코드", "행정동명"])
        match_type = "exact_name"
    elif not base_candidates.empty:
        candidates = base_candidates.drop_duplicates(subset=["행정동코드", "행정동명"])
        match_type = "base_name"
    else:
        candidates = keyword_table.iloc[0:0]
        match_type = "unmatched"

    candidate_names = candidates["행정동명"].tolist()
    candidate_codes = candidates["행정동코드"].astype(str).tolist()

    if len(candidates) == 1:
        status = "matched"
        dong_name = candidate_names[0]
        dong_code = candidate_codes[0]
    elif len(candidates) > 1:
        status = "ambiguous"
        dong_name = pd.NA
        dong_code = pd.NA
    else:
        status = "unmatched"
        dong_name = pd.NA
        dong_code = pd.NA

    return pd.Series(
        {
            "match_status": status,
            "match_type": match_type,
            "candidate_count": len(candidates),
            "candidate_dong_names": " | ".join(candidate_names),
            "candidate_dong_codes": " | ".join(candidate_codes),
            "행정동명": dong_name,
            "행정동코드": dong_code,
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    station = read_csv_with_fallback(STATION_PATH)
    station["station_id"] = normalize_station_id(station["대여소번호"])
    station = station[["station_id", "대여소번호", "대여소명", "주소", "위도", "경도"]].copy()

    mapping = read_csv_with_fallback(MAPPING_PATH)
    gangnam_mapping = mapping[mapping["행정동코드"].astype(str).str.startswith("11680")].copy()
    keyword_table = build_keyword_table(gangnam_mapping)

    assigned = station.apply(assign_candidates, axis=1, keyword_table=keyword_table)
    result = pd.concat([station, assigned], axis=1)

    result.to_csv(OUTPUT_DIR / "ddri_station_admin_dong_mapping_candidates.csv", index=False, encoding="utf-8-sig")
    result[result["match_status"] == "matched"].to_csv(
        OUTPUT_DIR / "ddri_station_admin_dong_mapping_matched.csv", index=False, encoding="utf-8-sig"
    )
    result[result["match_status"] == "ambiguous"].to_csv(
        OUTPUT_DIR / "ddri_station_admin_dong_mapping_ambiguous.csv", index=False, encoding="utf-8-sig"
    )
    result[result["match_status"] == "unmatched"].to_csv(
        OUTPUT_DIR / "ddri_station_admin_dong_mapping_unmatched.csv", index=False, encoding="utf-8-sig"
    )

    summary = (
        result.groupby(["match_status", "match_type"], dropna=False)
        .size()
        .rename("station_count")
        .reset_index()
        .sort_values(["match_status", "match_type"])
    )
    summary.to_csv(OUTPUT_DIR / "ddri_station_admin_dong_mapping_summary.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
