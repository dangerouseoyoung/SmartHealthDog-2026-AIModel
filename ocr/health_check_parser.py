import re
from datetime import date
from difflib import SequenceMatcher

from ocr.health_check_fields import (
    FIELD_ALIASES,
    MULTILINE_FIELDS,
    DATE_FIELDS
)


class HealthCheckParser:

    def __init__(self):
        self.aliases = FIELD_ALIASES

        # 병명 자체가 아니라 병명의 상태/구분을 나타내는 표현
        # 특정 질병명과는 무관함
        self.disease_meta_labels = [
            "임상적 추정",
            "임상적추정",
            "최종 진단",
            "최종진단"
        ]

    # =========================================================
    # 기본 텍스트 처리
    # =========================================================

    @staticmethod
    def normalize(text):

        if not text:
            return ""

        text = text.lower().strip()

        text = re.sub(
            r"[\s:：|·ㆍ_\-\[\]{}<>]",
            "",
            text
        )

        return text

    @staticmethod
    def clean_value(value):

        if value is None:
            return None

        value = re.sub(
            r"\s+",
            " ",
            value
        ).strip()

        value = value.strip(
            "|:："
        ).strip()

        if not value:
            return None

        return value

    # =========================================================
    # Alias 목록
    # =========================================================

    def alias_candidates(self):

        candidates = []

        for field, aliases in self.aliases.items():

            for alias in aliases:

                candidates.append(
                    (
                        field,
                        alias,
                        self.normalize(alias)
                    )
                )

        # 긴 라벨부터 검사
        candidates.sort(
            key=lambda x: len(x[2]),
            reverse=True
        )

        return candidates

    # =========================================================
    # 라벨 탐지
    # =========================================================

    def detect_label(self, text):

        normalized_text = self.normalize(
            text
        )

        if not normalized_text:
            return None, None

        # -----------------------------------------------------
        # 1. 정확 일치 또는 "라벨 + 값"
        # -----------------------------------------------------

        for (
            field,
            alias,
            normalized_alias
        ) in self.alias_candidates():

            if normalized_text == normalized_alias:
                return field, alias

            if (
                len(normalized_alias) >= 2
                and
                normalized_text.startswith(
                    normalized_alias
                )
            ):
                return field, alias

        # -----------------------------------------------------
        # 2. OCR이 긴 라벨을 조금 틀린 경우
        # -----------------------------------------------------

        best_match = None
        best_score = 0

        for (
            field,
            alias,
            normalized_alias
        ) in self.alias_candidates():

            # 짧은 단어 fuzzy matching 금지
            if len(normalized_alias) < 4:
                continue

            # 너무 긴 문장을 라벨로 오인하지 않도록 제한
            if (
                len(normalized_text)
                > len(normalized_alias) + 3
            ):
                continue

            score = SequenceMatcher(
                None,
                normalized_text,
                normalized_alias
            ).ratio()

            if score > best_score:

                best_score = score

                best_match = (
                    field,
                    alias
                )

        if (
            best_match is not None
            and best_score >= 0.82
        ):
            return best_match

        return None, None

    # =========================================================
    # OCR 블록 크기
    # =========================================================

    @staticmethod
    def width(block):

        return max(
            block["x2"]
            - block["x1"],
            1
        )

    @staticmethod
    def height(block):

        return max(
            block["y2"]
            - block["y1"],
            1
        )

    # =========================================================
    # 같은 행인지 판단
    # =========================================================

    def same_row(
        self,
        a,
        b
    ):

        tolerance = max(
            max(
                self.height(a),
                self.height(b)
            ) * 0.65,
            12
        )

        return (
            abs(
                a["center_y"]
                - b["center_y"]
            )
            <= tolerance
        )

    # =========================================================
    # Inline 값 추출
    # =========================================================

    def extract_inline_value(
        self,
        text,
        alias
    ):

        # alias 내부의 공백 차이 허용
        compact_alias = r"\s*".join(re.escape(part) for part in alias.split())

        match = re.search(
            compact_alias,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = text[
                match.end():
            ].strip(
                " :：|-"
            )

            if value:

                # 예:
                # 동물명(동물등록번호)
                # 여기서 (동물등록번호)를 동물명으로
                # 잘못 처리하는 것을 방지
                if (
                    value.startswith("(")
                    and
                    value.endswith(")")
                ):
                    return None

                return value

        # -----------------------------------------------------
        # 단일 단어 라벨
        # -----------------------------------------------------

        normalized_text = (
            self.normalize(text)
        )

        normalized_alias = (
            self.normalize(alias)
        )

        # "동물 명"처럼 OCR이 라벨 안에 공백을 넣은 경우에는
        # 라벨의 마지막 단어("명")를 값으로 착각하면 안 된다.
        if normalized_text == normalized_alias:
            return None

        if (
            normalized_text.startswith(
                normalized_alias
            )
            and
            len(alias.split()) == 1
        ):

            parts = text.split(
                maxsplit=1
            )

            if len(parts) == 2:

                value = (
                    parts[1]
                    .strip()
                )

                if value:
                    return value

        return None

    # =========================================================
    # 분리된 라벨 결합
    # =========================================================

    def merge_split_labels(
        self,
        blocks
    ):

        result = list(
            blocks
        )

        for i in range(
            len(blocks)
        ):

            a = blocks[i]

            for j in range(
                i + 1,
                min(
                    i + 3,
                    len(blocks)
                )
            ):

                b = blocks[j]

                # 완성된 라벨과 다른 행의 텍스트를 붙이면
                # "특징 별명" 같은 가짜 inline 값이 만들어진다.
                if self.detect_label(a["text"])[0] is not None:
                    continue
                if self.detect_label(b["text"])[0] is not None:
                    continue

                # 다른 표 행 간 결합 금지. 분리된 글자 라벨만 결합한다.
                if not self.same_row(a, b):
                    continue

                vertical_distance = abs(
                    a["center_y"]
                    - b["center_y"]
                )

                if vertical_distance > 45:
                    continue

                horizontal_gap = (
                    b["x1"]
                    - a["x2"]
                )

                if horizontal_gap > 100:
                    continue

                combined = (
                    a["text"]
                    + " "
                    + b["text"]
                )

                field, _ = (
                    self.detect_label(
                        combined
                    )
                )

                if field is None:
                    continue

                x1 = min(
                    a["x1"],
                    b["x1"]
                )

                y1 = min(
                    a["y1"],
                    b["y1"]
                )

                x2 = max(
                    a["x2"],
                    b["x2"]
                )

                y2 = max(
                    a["y2"],
                    b["y2"]
                )

                result.append({
                    "text": combined,

                    "confidence": min(
                        a["confidence"],
                        b["confidence"]
                    ),

                    "bbox": [],

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "center_x": int(
                        (x1 + x2) / 2
                    ),

                    "center_y": int(
                        (y1 + y2) / 2
                    )
                })

        result.sort(
            key=lambda block: (
                block["center_y"],
                block["center_x"]
            )
        )

        return result

    # =========================================================
    # 라벨 목록
    # =========================================================

    def find_labels(
        self,
        blocks
    ):

        labels = []

        for index, block in enumerate(
            blocks
        ):

            field, alias = (
                self.detect_label(
                    block["text"]
                )
            )

            if field is None:
                continue

            labels.append({
                "index": index,
                "field": field,
                "alias": alias,
                "block": block
            })

        return labels

    def is_label(
        self,
        block
    ):

        field, _ = (
            self.detect_label(
                block["text"]
            )
        )

        return field is not None

    # =========================================================
    # 같은 행 오른쪽 값
    # =========================================================

    def values_on_right(
        self,
        label_block,
        blocks
    ):

        candidates = []

        # 2열 표에서 오른쪽의 *다른 필드 라벨*을 넘어서
        # 값을 가져오지 않도록 칸 경계를 찾는다.
        other_labels_to_right = [
            b["x1"] for b in blocks
            if b is not label_block
            and self.is_label(b)
            and self.same_row(label_block, b)
            and b["center_x"] > label_block["center_x"]
        ]
        right_boundary = min(other_labels_to_right) if other_labels_to_right else float("inf")

        for block in blocks:

            if block is label_block:
                continue

            if self.is_label(
                block
            ):
                continue

            if not self.same_row(
                label_block,
                block
            ):
                continue

            # 반드시 라벨 오른쪽
            if (
                block["center_x"]
                <=
                label_block["center_x"]
            ):
                continue

            if block["center_x"] >= right_boundary:
                continue

            distance = max(
                0,
                block["x1"]
                - label_block["x2"]
            )

            candidates.append(
                (
                    distance,
                    block
                )
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]["center_x"]
            )
        )

        return [
            item[1]
            for item
            in candidates
        ]

    # =========================================================
    # 다음 라벨 Y 좌표
    # =========================================================

    def next_label_y(
        self,
        current_label,
        labels
    ):

        current_y = (
            current_label[
                "block"
            ][
                "center_y"
            ]
        )

        following = []

        for label in labels:

            y = (
                label[
                    "block"
                ][
                    "center_y"
                ]
            )

            if y > current_y + 10:
                following.append(
                    y
                )

        if not following:
            return None

        return min(
            following
        )

    # =========================================================
    # 라벨 아래 값
    # =========================================================

    def values_below(
        self,
        label,
        blocks,
        labels
    ):

        label_block = (
            label["block"]
        )

        start_y = (
            label_block[
                "center_y"
            ]
        )

        end_y = (
            self.next_label_y(
                label,
                labels
            )
        )

        if end_y is None:

            end_y = (
                start_y
                + max(
                    self.height(
                        label_block
                    ) * 5,
                    180
                )
            )

        candidates = []

        for block in blocks:

            if block is label_block:
                continue

            if self.is_label(
                block
            ):
                continue

            cy = block[
                "center_y"
            ]

            if cy <= start_y + 5:
                continue

            if cy >= end_y - 5:
                continue

            candidates.append(
                block
            )

        candidates.sort(
            key=lambda block: (
                block["center_y"],
                block["center_x"]
            )
        )

        return candidates

    # =========================================================
    # 일반 단일 값
    # =========================================================

    def extract_single_value(
        self,
        label,
        blocks,
        labels
    ):

        block = (
            label["block"]
        )

        alias = (
            label["alias"]
        )

        # -----------------------------------------------------
        # 1. 같은 OCR 블록
        # -----------------------------------------------------

        inline = (
            self.extract_inline_value(
                block["text"],
                alias
            )
        )

        if inline:

            field, _ = (
                self.detect_label(
                    inline
                )
            )

            if field is None:
                return inline

        # -----------------------------------------------------
        # 2. 같은 행 오른쪽
        # -----------------------------------------------------

        right = (
            self.values_on_right(
                block,
                blocks
            )
        )

        if right:

            nearest = (
                right[0]
            )

            gap = max(
                0,
                nearest["x1"]
                - block["x2"]
            )

            # 다른 표 칸을 값으로 가져오는 것 방지
            max_gap = max(
                self.width(block)
                * 3,
                250
            )

            if gap <= max_gap:
                return nearest[
                    "text"
                ]

        # 단일 값은 함부로 다음 행에서
        # 가져오지 않는다.
        return None

    # =========================================================
    # 병명의 보조 라벨인지 검사
    # =========================================================

    def is_disease_meta_label(
        self,
        text
    ):

        normalized = (
            self.normalize(
                text
            )
        )

        for meta in (
            self.disease_meta_labels
        ):

            normalized_meta = (
                self.normalize(
                    meta
                )
            )

            if (
                normalized_meta
                in normalized
            ):
                return True

        return False

    # =========================================================
    # 여러 줄 값
    # =========================================================

    def extract_multiline_value(
        self,
        label,
        blocks,
        labels
    ):

        block = (
            label["block"]
        )

        alias = (
            label["alias"]
        )

        field = (
            label["field"]
        )

        values = []

        # -----------------------------------------------------
        # 1. 같은 OCR 블록
        # -----------------------------------------------------

        inline = (
            self.extract_inline_value(
                block["text"],
                alias
            )
        )

        if inline:

            detected_field, _ = (
                self.detect_label(
                    inline
                )
            )

            if detected_field is None:

                if not (
                    field
                    == "disease_name"
                    and
                    self.is_disease_meta_label(
                        inline
                    )
                ):
                    values.append(
                        inline
                    )

        # -----------------------------------------------------
        # 2. 같은 행 오른쪽
        # -----------------------------------------------------

        right = (
            self.values_on_right(
                block,
                blocks
            )
        )

        for candidate in right:

            text = (
                candidate[
                    "text"
                ]
            )

            if (
                field
                == "disease_name"
                and
                self.is_disease_meta_label(
                    text
                )
            ):
                continue

            if text not in values:

                values.append(
                    text
                )

        # -----------------------------------------------------
        # 3. 다음 주요 라벨 전까지
        # -----------------------------------------------------

        below = (
            self.values_below(
                label,
                blocks,
                labels
            )
        )

        for candidate in below:

            text = (
                candidate[
                    "text"
                ]
            )


            # 문서 하단의 독립적인 날짜 블록 제외
            if self.normalize_date(text) and re.fullmatch(
                r"\s*20\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\s*",
                text
            ):
                continue


            detected_field, _ = (
                self.detect_label(
                    text
                )
            )

            if detected_field is not None:
                continue

            if (
                field
                == "disease_name"
                and
                self.is_disease_meta_label(
                    text
                )
            ):
                continue

            # 여러 줄 필드에서도 왼쪽의 다른 라벨이나
            # 다른 표 행의 날짜를 값으로 가져오지 않게 한다.
            if candidate["center_x"] <= block["center_x"] + 8:
                continue
            if field == "disease_name" and self.normalize_date(text):
                continue

            if text not in values:

                values.append(
                    text
                )

        if not values:
            return None

        return " ".join(
            values
        )

    # =========================================================
    # NOTES / 비고
    # =========================================================

    def extract_notes(
        self,
        label,
        blocks
    ):

        """
        비고/그 밖의 사항은 문서 하단의
        법률 안내문을 잘못 가져오기 쉬움.

        따라서 같은 OCR 박스 또는
        같은 행 오른쪽 값만 사용한다.
        """

        block = (
            label["block"]
        )

        alias = (
            label["alias"]
        )

        # 같은 블록
        inline = (
            self.extract_inline_value(
                block["text"],
                alias
            )
        )

        if inline:
            return inline

        # 같은 행 오른쪽
        right = (
            self.values_on_right(
                block,
                blocks
            )
        )

        if not right:
            return None

        values = []

        for candidate in right:

            text = candidate[
                "text"
            ]

            if text not in values:

                values.append(
                    text
                )

        if not values:
            return None

        return " ".join(
            values
        )

    # =========================================================
    # 날짜 정규화
    # =========================================================

    @staticmethod
    def normalize_date(
        text
    ):

        if not text:
            return None

        # -----------------------------------------------------
        # 2025년 10월 31일
        # -----------------------------------------------------

        match = re.search(
            r"(20\d{2})\s*년\s*"
            r"(\d{1,2})\s*월\s*"
            r"(\d{1,2})\s*일",
            text
        )

        if match:

            year, month, day = (
                match.groups()
            )

            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return None

        # -----------------------------------------------------
        # 2025.10.31
        # 2025-10-31
        # 2025/10/31
        # -----------------------------------------------------

        match = re.search(
            r"(20\d{2})\s*"
            r"[./\-]\s*"
            r"(\d{1,2})\s*"
            r"[./\-]\s*"
            r"(\d{1,2})",
            text
        )

        if match:

            year, month, day = (
                match.groups()
            )

            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return None

        # Google Vision이 구분자 없이 읽는 YYYYMMDD (예: 20201028)
        match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", text)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None

        return None

    # =========================================================
    # 날짜 필드
    # =========================================================

    def extract_date(
        self,
        label,
        blocks,
        labels
    ):

        block = (
            label["block"]
        )

        # 입원/퇴원일은 범위일 수 있으므로 단일 날짜로 잘라내지 않는다.
        if label["field"] == "hospitalization_discharge_date":
            right = self.values_on_right(block, blocks)
            below = self.values_below(label, blocks, labels)
            candidate = " ".join([block["text"]] + [b["text"] for b in right] + [b["text"] for b in below[:5]])
            tokens = re.findall(
                r"20\d{2}\s*(?:년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2})",
                candidate,
            )
            dates = []
            for token in tokens:
                normalized = self.normalize_date(token)
                if normalized and normalized not in dates:
                    dates.append(normalized)
            if dates:
                return " ~ ".join(dates[:2])

        # -----------------------------------------------------
        # 1. 같은 OCR 블록
        # -----------------------------------------------------

        inline = (
            self.extract_inline_value(
                block["text"],
                label["alias"]
            )
        )

        if inline:

            date = (
                self.normalize_date(
                    inline
                )
            )

            if date:
                return date

        # -----------------------------------------------------
        # 2. 같은 행 오른쪽
        # -----------------------------------------------------

        right = (
            self.values_on_right(
                block,
                blocks
            )
        )

        if right:

            text = " ".join(
                item["text"]
                for item
                in right
            )

            date = (
                self.normalize_date(
                    text
                )
            )

            if date:
                return date

        # -----------------------------------------------------
        # 3. 아래쪽 날짜 조각
        # -----------------------------------------------------

        below = (
            self.values_below(
                label,
                blocks,
                labels
            )
        )

        if below:

            text = " ".join(
                item["text"]
                for item
                in below[:5]
            )

            date = (
                self.normalize_date(
                    text
                )
            )

            if date:
                return date

        return None

    # =========================================================
    # 문서 하단 작성일 탐색
    # =========================================================

    def find_document_date(
        self,
        blocks,
        excluded_dates=None
    ):

        """
        '작성일'이라는 라벨 없이

        2026년 9월 28일

        처럼 문서 하단에 날짜만 있는
        진단서도 처리한다.
        """

        if not blocks:
            return None

        max_y = max(
            block["center_y"]
            for block in blocks
        )

        # 문서 아래쪽 40%
        threshold = (
            max_y * 0.60
        )

        bottom_blocks = [
            block
            for block in blocks
            if (
                block["center_y"]
                >= threshold
            )
        ]

        # -----------------------------------------------------
        # 비슷한 Y 좌표끼리 행 생성
        # -----------------------------------------------------

        rows = []

        for block in bottom_blocks:

            placed = False

            for row in rows:

                if abs(
                    row[0]["center_y"]
                    - block["center_y"]
                ) <= 25:

                    row.append(
                        block
                    )

                    placed = True
                    break

            if not placed:

                rows.append(
                    [block]
                )

        dates = []

        for row in rows:

            row.sort(
                key=lambda block:
                block["center_x"]
            )

            text = " ".join(
                block["text"]
                for block
                in row
            )

            date = (
                self.normalize_date(
                    text
                )
            )

            if date:

                average_y = sum(
                    block["center_y"]
                    for block
                    in row
                ) / len(row)

                dates.append(
                    (
                        average_y,
                        date
                    )
                )

        if not dates:
            return None

        
        # 이미 다른 날짜 필드에서 사용한 날짜는
        # 작성일 후보에서 제외한다.
        excluded_dates = set(excluded_dates or [])

        dates = [
            (y, date)
            for y, date in dates
            if date not in excluded_dates
        ]

        if not dates:
            return None

        # 남은 날짜 중 가장 아래쪽 날짜 선택
        dates.sort(key=lambda item: item[0])

        return dates[-1][1]

    # =========================================================
    # 등록번호
    # =========================================================

    @staticmethod
    def extract_registration_number(
        value
    ):

        if not value:
            return None

        match = re.search(
            r"\b\d[\d\- ]{8,18}\d\b",
            value
        )

        if not match:
            return None

        return re.sub(
            r"\D",
            "",
            match.group()
        )

    # =========================================================
    # 진단서 표 OCR 오인식 복원 (문서 내용 자체를 만들지 않음)
    # =========================================================

    def recover_template_labels(self, original_blocks):
        """일부 OCR 오타를 *주변 라벨과 표 위치가 확실할 때만* 복원.

        원본 OCR 데이터는 수정하지 않는다. 임의의 진단명/날짜는 만들지 않는다.
        """
        blocks = [dict(b) for b in original_blocks]
        all_fields = [(self.detect_label(b["text"])[0], b) for b in blocks]

        def first_y(field):
            matches = [b["center_y"] for f, b in all_fields if f == field]
            return min(matches) if matches else None

        features_y = first_y("features")
        diag_y = first_y("diagnosis_date")
        if features_y is None or diag_y is None:
            return blocks

        # 이 진단서처럼 특징 → 병명 → 발병일 → 진단일 순서가 확인될 때만
        # "별명"(병명 오인식)을 보정한다.
        disease_blocks = [b for f, b in all_fields if f == "disease_name"]
        if not disease_blocks:
            malformed_onset = any(
                self.normalize(b["text"]).startswith("별명(")
                for b in blocks
            )
            has_date_row = any(
                self.normalize(b["text"]) == "연월일"
                and features_y < b["center_y"] < diag_y
                for b in blocks
            )
            if malformed_onset and has_date_row:
                for b in blocks:
                    if (
                        self.normalize(b["text"]) == "별명"
                        and features_y + 10 < b["center_y"] < diag_y - 12
                    ):
                        b["text"] = "병명"
                        break

        # "발병(발견) 연월일"이 두 블록으로 손상되어 인식된 경우.
        has_onset = any(self.detect_label(b["text"])[0] == "onset_date" for b in blocks)
        if not has_onset:
            for b in blocks:
                if not (
                    self.normalize(b["text"]) == "연월일"
                    and features_y < b["center_y"] < diag_y
                ):
                    continue
                near_malformed = any(
                    self.normalize(x["text"]).startswith("별명(")
                    and abs(x["center_y"] - b["center_y"]) <= 22
                    for x in blocks if x is not b
                )
                near_date = any(
                    self.normalize_date(x["text"])
                    and abs(x["center_y"] - b["center_y"]) <= 18
                    and x["center_x"] > b["center_x"]
                    for x in blocks if x is not b
                )
                if near_malformed and near_date:
                    b["text"] = "발병 연월일"
                    break

        # 진단서 하단 "그 밖의 사항"을 "그림 사항"으로 잘못 읽는 경우.
        if not any(self.detect_label(b["text"])[0] == "notes" for b in blocks):
            for b in blocks:
                if (
                    self.normalize(b["text"]) == "그림사항"
                    and b["center_y"] > diag_y + 30
                ):
                    b["text"] = "그 밖의 사항"
                    break

        # 예후 소견은 긴 내용과 왼쪽 라벨이 한 OCR 블록에 합쳐지기도 함.
        # 예후 라벨이 없어도 병원 소견 본문은 원문 블록 그대로 보존한다.
        if not any(self.detect_label(b["text"])[0] == "prognosis" for b in blocks):
            note_ys = [
                b["center_y"] for b in blocks
                if self.detect_label(b["text"])[0] == "notes"
                and b["center_y"] > diag_y
            ]
            if note_ys:
                notes_y = min(note_ys)
                candidates = [
                    b for b in blocks
                    if diag_y + 12 < b["center_y"] < notes_y - 12
                    and len(b["text"]) >= 24
                    and ("검사" in b["text"] or "의심" in b["text"])
                ]
                if candidates:
                    main = min(candidates, key=lambda b: b["center_y"])
                    synthetic = {
                        "text": "예후 소견",
                        "confidence": 0.0,
                        "x1": 10, "x2": 65,
                        "y1": main["center_y"] - 8,
                        "y2": main["center_y"] + 8,
                        "center_x": 37,
                        "center_y": main["center_y"],
                        "bbox": [],
                    }
                    blocks.append(synthetic)

        blocks.sort(key=lambda b: (b["center_y"], b["center_x"]))
        return blocks

    # =========================================================
    # 전체 Parsing
    # =========================================================

    def parse(
        self,
        ocr_result
    ):

        original_blocks = (
            ocr_result[
                "blocks"
            ]
        )

        blocks = (
            self.merge_split_labels(
                self.recover_template_labels(original_blocks)
            )
        )

        labels = (
            self.find_labels(
                blocks
            )
        )

        result = {
            field: None
            for field
            in self.aliases.keys()
        }

        # -----------------------------------------------------
        # 필드별 값 추출
        # -----------------------------------------------------

        for label in labels:

            field = (
                label["field"]
            )

            # 이미 값을 얻은 경우
            if (
                result[field]
                is not None
            ):
                continue

            # 날짜
            if field in DATE_FIELDS:

                value = (
                    self.extract_date(
                        label,
                        blocks,
                        labels
                    )
                )

            # 비고
            elif field == "notes":

                value = (
                    self.extract_notes(
                        label,
                        blocks
                    )
                )

            # 여러 줄
            elif field in MULTILINE_FIELDS:

                value = (
                    self.extract_multiline_value(
                        label,
                        blocks,
                        labels
                    )
                )

            # 일반 단일 값
            else:

                value = (
                    self.extract_single_value(
                        label,
                        blocks,
                        labels
                    )
                )

            result[field] = (
                self.clean_value(
                    value
                )
            )

        # -----------------------------------------------------
        # 작성일 라벨이 없는 문서
        # -----------------------------------------------------

        
        if result.get("document_date") is None:

            # 이미 추출된 날짜 필드의 값 수집
            excluded_dates = {
                result[field]
                for field in DATE_FIELDS
                if field != "document_date"
                and result.get(field) is not None
            }

            result["document_date"] = (
                self.find_document_date(
                    original_blocks,
                    excluded_dates=excluded_dates
                )
            )


        # -----------------------------------------------------
        # 동물명 안에 등록번호가 섞여 있는 경우
        # -----------------------------------------------------

        animal_name = (
            result.get(
                "animal_name"
            )
        )

        if animal_name:

            registration = (
                self.extract_registration_number(
                    animal_name
                )
            )

            if registration:

                if (
                    result.get(
                        "registration_number"
                    )
                    is None
                ):

                    result[
                        "registration_number"
                    ] = (
                        registration
                    )

        return result