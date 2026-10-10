# MyTrader Streamlit MVP

## 실행
Windows PowerShell에서 이 폴더로 이동한 뒤:

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1

접속 주소: http://localhost:8501

## 입력
- 종목코드: 6자리 숫자
- 종목명: 엑셀 시트명과 동일
- 종료일: 어제까지
- 엑셀 필수 컬럼: 분기, 목표가, 영업이익
- 분기 형식 예: 2026.1Q

종목명과 종목코드의 일치 여부는 자동 검증하지 않으므로
반드시 확인하고 입력하세요.

## 주의
- 원본 계산 클래스들을 재사용한 MVP입니다.
- 주간/월간 지표의 미래 데이터 참조 위험이 있습니다.
- 백테스트 결과는 검증 전 참고용입니다.
- 장중 네이버 데이터는 분석에 사용하지 않습니다.
- 분석 결과는 세션에 보관하며 새 세션에서는 사라질 수 있습니다.
- pykrx/KRX 데이터 서비스 상태에 따라 조회가 실패할 수 있습니다.
- 원본 코드를 보관한 폴더를 정적 웹 서버로 공개하지 마세요.
- 회원 관리, 외부 공개용 인증, 주문 실행은 구현하지 않았습니다.
