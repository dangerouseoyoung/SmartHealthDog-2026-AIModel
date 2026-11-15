"""Recreates only text/center positions the user printed; no credentials/network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.health_check_parser import HealthCheckParser


def block(text, cx, cy, width=None):
    width = width if width is not None else max(18, len(text)*5)
    x1 = int(cx - width / 2)
    x2 = int(cx + width / 2)
    return dict(text=text, confidence=.83, x1=x1, x2=x2, y1=cy-7, y2=cy+7,
                center_x=cx, center_y=cy, bbox=[])


def fixture_blocks():
    return [block(*v) for v in [
        ('진단서',237,16),
        ('성명',88,54),
        ('동물 소유자',39,63),
        ('( 관리인 )',39,80),
        ('주소',91,86),
        ('사육 장소',42,115),
        ('품중',265,141),
        ('Pug',304,141),
        ('종류 Canine',113,145),
        ('Castrated Male',327,168),
        ('정벌',264,170),
        ('동물 명',96,175),
        ('동물 의',48,184),
        ('표시',48,195),
        ('모색',265,198),
        ('연령 16 년 3 개월',123,201),
        ('특징',98,230),
        ('심장 비대 , 폐 침윤',113,258),
        ('별명',54,259),
        ('별명 ( 일산 )',54,282),
        ('20201028',104,286),
        ('연월일',55,293),
        ('진단 연월일 20201028',80,316),
        ('루소 엑스레이 검사 상 폐 침윤 이 확인 되며 폐수종 , 폐렴 , 폐점 의심 됩니다 . 소견',193,345,260),
        ('1028 일 진료 에 대한 진단서 발급 원하실',157,366,190),
        ('그림 사항',59,390),
        ('수사법 제 12 조 및 같은 법 시',180,438),
    ]]


def test_sample_layout_without_google_api():
    parser=HealthCheckParser()
    result=parser.parse({'blocks':fixture_blocks()})
    assert len(result)==17
    assert result['species']=='Canine'
    assert result['breed']=='Pug'
    assert result['gender']=='Castrated Male'
    assert result['animal_name'] is None
    assert result['features'] is None
    assert result['age']=='16 년 3 개월'
    assert '심장 비대' in (result['disease_name'] or '')
    assert '폐 침윤' in (result['disease_name'] or '')
    assert '별명' not in (result['disease_name'] or '')
    assert '20201028' not in (result['disease_name'] or '')
    assert result['onset_date']=='2020-10-28'
    assert result['diagnosis_date']=='2020-10-28'
    assert '엑스레이' in (result['prognosis'] or '')
    assert '발급' in (result['prognosis'] or '')
    assert result['notes'] is None
    assert result['document_date'] is None


def test_no_hallucinated_disease_from_unrelated_nickname():
    parser=HealthCheckParser()
    blocks=[block('동물명',90,20),block('별명',50,60),block('호두',140,60),block('특징',90,200)]
    result=parser.parse({'blocks':blocks})
    assert result['disease_name'] is None


def test_compact_date_validation():
    p=HealthCheckParser()
    assert p.normalize_date('20201028')=='2020-10-28'
    assert p.normalize_date('20260230') is None
    assert p.normalize_date('12320201028000') is None
