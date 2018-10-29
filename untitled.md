## 원래 일정
- CrowdWorks
    - 이번주 목에 진행 예상이었으나 crowdworks의 사정으로 늦어짐
- 원래의 목표
    - 3개의 모델에 대해 시도해보고 각 보고 최종 모델 선정 & reranking
    - SMN, WS, LSTM with Attention
    
## 진행 상황
- 모델 구현 결과
    - Sequential Matching Network: 구현 완료, 어느 정도의 수준은 되지만 최고 모델에 못미침 (5등 정도)
    - WS with Seq2Seq: Seq2Seq 계속된 OOM문제 및 드라이버 오류 문제로 실패. 그냥 Sampled softmax loss버리고 vocab 수 줄여서 다시 돌릴 예정
    - Attention 구조: transformer구조는 시도 안해보고 단순 lstm + attention 구조 시도해볼 예정(은 아직 짜지도 않음 ㅎㅎ)
    - 추가로 기본 모델에 대한 실험을 좀 더 해봄
        - 기존 모델에서 약간 변형한 결과 성능이 다소 올라감
        - LSTM+CNN(to rnn outputs): 성능이 별로임
- 최종 일정
    - 다음주까지 모델 선정 및 최적화
    - 그 다음주까지 reranking까지 마무리하고 다음 분기 시작
- Reranking 구조 짜고 피쳐 미리 짜놓기
    - 둘다 동시에 하니깐 집중이 안됨 ㅠ
    - 요일 정해서 해야할 듯
- 라인으로 보는 속마음 회의에 상준님과 함께 계속 참석 예정
    
## 다음주에 할 일
- 크라우드웍스 데이터 받기 전까지 reranking 구조 마무리
- 모델에 대한 추가적인 미련은 버리고, 위의 모델만 빨리 마무리하고 결과 비교
