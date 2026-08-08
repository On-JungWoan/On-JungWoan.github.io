# 일본 여행 페이지 데이터 편집 가이드

여행 페이지에 표시되는 일정, 지도 장소, 예약 지갑, 체크리스트의 단일 원본은 [`_data/japan_trip.yml`](../../_data/japan_trip.yml)입니다. 이 문서는 사용법만 설명하며, 여기에 일정을 적어도 페이지에는 반영되지 않습니다.

`projects/japan_trip/index.html`과 `assets/js/trip-data.js`는 YAML 데이터를 화면과 JavaScript용 데이터로 변환하므로 평소에는 수정하지 않아도 됩니다.

## 가장 자주 수정할 곳

### 일정 카드

`days`에서 날짜를 찾고 그 아래 `events`를 수정합니다.

```yml
- id: 2
  shortLabel: '8.18'
  title: 벳푸 핵심 관광
  events:
  - id: d2-new-place
    day: 2
    start: '2026-08-18T15:00:00+09:00'
    end: '2026-08-18T16:30:00+09:00'
    departAt: '2026-08-18T14:30:00+09:00'
    title: 새 장소로 이동
    destination: 새 장소
    timeLabel: '15:00'
    timeSuffix: "—16:30"
    typeLabel: Sightseeing
    cardTitle: 새 장소
    stopIds:
    - new-place
    mapLabel: 새 장소
    mapLink: true
    description: 카드에 표시할 설명
```

- `id`: 페이지 전체에서 겹치지 않는 영문 ID입니다. 기존 ID는 바꾸지 않는 편이 안전합니다.
- `start`, `end`: 당일 모드가 지난 일정·현재 일정·다음 일정을 판단하는 기준입니다.
- `departAt`: 상단에 표시되는 출발 카운트다운 기준 시각입니다.
- `timeLabel`, `timeSuffix`: 카드에 실제로 보이는 시각입니다.
- `compact: true`: 위치 버튼이 없는 간단한 카드로 표시합니다.
- `optional: true`: 선택 일정 스타일을 적용합니다.
- `descriptionHtml`: 굵은 글자처럼 HTML 표현이 꼭 필요할 때만 `description` 대신 사용합니다.

날짜와 시각에는 항상 한국·일본 시간대인 `+09:00`을 붙입니다.

### 길찾기와 지도 장소

일정에서 새 장소를 사용하려면 `stops`에도 같은 ID의 장소를 추가합니다.

```yml
stops:
- id: new-place
  name: 새 장소
  lat: 33.000000
  lng: 131.000000
  days:
  - 2
  label: 지도 핀에 표시할 짧은 설명
  query: Google 지도에서 검색할 정확한 장소명
```

그리고 해당 Day의 `stops`와 이벤트의 `stopIds`에 `new-place`를 넣습니다. 좌표는 지도 핀, `query`는 Google 지도 링크에 쓰입니다.

출발지와 목적지가 채워진 길찾기 버튼이 필요하면 이벤트에 아래를 추가합니다.

```yml
directions:
  origin: previous-place
  destination: new-place
  mode: driving
```

`origin`과 `destination`은 반드시 `stops`에 존재하는 ID여야 합니다. `mode`는 `driving`, `transit`, `walking`, `bicycling` 중 하나를 사용합니다.

Day의 전체 경로는 `routeType`으로 정합니다.

- `schematic`: Day의 `stops` 순서대로 직선을 연결합니다.
- `geojson`: `routeFile`에 지정한 GeoJSON 경로 파일을 사용합니다.

GeoJSON을 새로 만들지 않았다면 `schematic`을 사용하면 됩니다.

### 예약 정보 지갑

`walletItems`에서 항공, 숙소, 렌터카, 티켓을 수정하거나 같은 형식으로 항목을 추가합니다.

```yml
- id: new-ticket
  type: ticket
  kicker: TICKET / 0818
  title: 입장권 이름
  summary: 2026.08.18
  fields:
  - label: 예약번호
    value: '1234567890'
    copyable: true
  - label: E-ticket
    value: "/projects/japan_trip/docs/ticket.pdf"
    href: "/projects/japan_trip/docs/ticket.pdf"
    linkLabel: PDF 열기
    copyable: false
```

- 예약번호처럼 앞자리 0이 중요하거나 숫자가 긴 값은 따옴표로 감쌉니다.
- `copyable: true`면 복사 버튼이 생깁니다.
- `href`가 있으면 `linkLabel` 이름의 링크가 생깁니다.
- PDF는 `projects/japan_trip/docs/`에 넣고 `/projects/japan_trip/docs/파일명.pdf` 형식의 루트 상대 경로를 사용합니다.
- 아직 모르는 값은 `value: 미입력`으로 두면 복사 버튼이 비활성화됩니다.

### 출발 전 체크리스트

페이지 하단 체크 항목은 `checklist`에서 관리합니다.

```yml
checklist:
- id: weather
  title: 날씨·산길
  description: 태풍, 폭우와 도로 상황
```

`id`를 바꾸면 브라우저에 저장된 기존 체크 상태와 연결이 끊기므로 제목이나 설명만 수정하는 편이 좋습니다.

## 날짜를 추가할 때

1. `days`에 기존 Day 하나와 같은 구조로 새 Day를 추가합니다.
2. `id`, 날짜 표시, 제목, 색상, 이동 방식, `events`를 채웁니다.
3. 새 장소가 있으면 `stops`에 등록하고 Day와 이벤트에서 그 ID를 참조합니다.
4. 전체 여행 기간이 늘어나면 맨 위 `trip.start`와 `trip.end`도 수정합니다.

개요 카드, 날짜 탭, 지도 범례, 상세 타임라인, 당일 모드는 이 데이터를 기준으로 자동 생성됩니다.

## 수정 후 확인

로컬에서 아래 명령으로 Jekyll 빌드를 확인합니다.

```powershell
bundle exec jekyll build --limit_posts 1
```

YAML 들여쓰기가 어긋나면 빌드가 실패합니다. 탭 대신 공백을 사용하고, 항목의 `-`와 하위 필드 들여쓰기를 기존 데이터와 맞춰 주세요.
