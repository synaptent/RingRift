# Event Wiring Diagram

**Last Updated**: January 7, 2026

This document provides visual diagrams of the RingRift event system using Mermaid syntax for GitHub rendering.

## Complete Training Pipeline

```mermaid
flowchart LR
    subgraph Selfplay["Selfplay Layer"]
        SP[SelfplayRunner]
        GPU[GPU Parallel Games]
    end

    subgraph DataPipeline["Data Pipeline"]
        DC[DataConsolidation]
        EX[AutoExport]
        NPZ[NPZ Combination]
    end

    subgraph Training["Training Layer"]
        TC[TrainingCoordinator]
        TT[TrainingTrigger]
    end

    subgraph Evaluation["Evaluation Layer"]
        ED[EvaluationDaemon]
        GG[GameGauntlet]
    end

    subgraph Promotion["Promotion Layer"]
        AP[AutoPromotion]
        DD[Distribution]
    end

    SP -->|SELFPLAY_COMPLETE| DC
    SP -->|NEW_GAMES_AVAILABLE| EX
    GPU -->|SELFPLAY_COMPLETE| DC

    DC -->|DATA_SYNC_COMPLETED| EX
    EX -->|TRAINING_THRESHOLD_REACHED| TT

    TT -->|triggers| TC
    TC -->|TRAINING_COMPLETED| ED

    ED -->|EVALUATION_COMPLETED| AP
    GG -->|EVALUATION_COMPLETED| AP

    AP -->|MODEL_PROMOTED| DD
    DD -->|MODEL_DISTRIBUTION_COMPLETE| SP
```

## Feedback Loops

```mermaid
flowchart TB
    subgraph Loop1["Quality → Training"]
        QM[QualityMonitor] -->|QUALITY_ASSESSMENT| FLC1[FeedbackLoopController]
        FLC1 -->|adjusts| TT1[TrainingTrigger]
    end

    subgraph Loop2["Elo Velocity → Selfplay"]
        ES[EloService] -->|ELO_UPDATED| SS[SelfplayScheduler]
        SS -->|reallocates| SP2[SelfplayRunner]
    end

    subgraph Loop3["Regression → Curriculum"]
        RD[RegressionDetector] -->|REGRESSION_DETECTED| FLC3[FeedbackLoopController]
        FLC3 -->|CURRICULUM_EMERGENCY_UPDATE| CI[CurriculumIntegration]
    end

    subgraph Loop4["Loss Anomaly → Exploration"]
        TC4[TrainingCoordinator] -->|LOSS_ANOMALY_DETECTED| FLC4[FeedbackLoopController]
        FLC4 -->|EXPLORATION_BOOST| SP4[SelfplayRunner]
    end

    subgraph Loop5["Curriculum → Weights"]
        CI5[CurriculumIntegration] -->|CURRICULUM_REBALANCED| SS5[SelfplayScheduler]
        SS5 -->|updates allocation| SP5[SelfplayRunner]
    end
```

## Event Categories by Flow

```mermaid
graph TD
    subgraph Training Events
        TS[TRAINING_STARTED]
        TC[TRAINING_COMPLETED]
        TF[TRAINING_FAILED]
        TP[TRAINING_PROGRESS]
        TBQ[TRAINING_BLOCKED_BY_QUALITY]
    end

    subgraph Selfplay Events
        SC[SELFPLAY_COMPLETE]
        NGA[NEW_GAMES_AVAILABLE]
        STU[SELFPLAY_TARGET_UPDATED]
        SAU[SELFPLAY_ALLOCATION_UPDATED]
    end

    subgraph Evaluation Events
        ES[EVALUATION_STARTED]
        EC[EVALUATION_COMPLETED]
        EF[EVALUATION_FAILED]
        EB[EVALUATION_BACKPRESSURE]
    end

    subgraph Promotion Events
        MP[MODEL_PROMOTED]
        PR[PROMOTION_ROLLED_BACK]
        MDS[MODEL_DISTRIBUTION_STARTED]
        MDC[MODEL_DISTRIBUTION_COMPLETE]
    end

    subgraph Health Events
        HO[HOST_OFFLINE]
        HON[HOST_ONLINE]
        PCH[P2P_CLUSTER_HEALTHY]
        PCU[P2P_CLUSTER_UNHEALTHY]
        LE[LEADER_ELECTED]
    end

    SC --> TC
    NGA --> TP
    TC --> ES
    EC --> MP
    MP --> MDC
    MDC --> SC
```

## Circuit Breaker Cascade

```mermaid
flowchart TD
    subgraph Tier1["Tier 1: Per-Operation"]
        OB1[OperationCB]
    end

    subgraph Tier2["Tier 2: Per-Node"]
        NB2[NodeCB]
    end

    subgraph Tier3["Tier 3: Per-Transport"]
        TB3[TransportCB]
    end

    subgraph Tier4["Tier 4: Cascade/Cluster"]
        CB4[CascadeBreakerManager]
    end

    OB1 -->|3 failures| NB2
    NB2 -->|5 failures| TB3
    TB3 -->|node unhealthy| CB4

    CB4 -->|P2P_RECOVERY_NEEDED| Recovery[P2PRecoveryDaemon]
    CB4 -->|ESCALATION_TIER_CHANGED| Monitor[HealthCoordinator]
```

## Data Sync Flow

```mermaid
flowchart LR
    subgraph Source["Data Sources"]
        SP[SelfplayRunner]
        GPU[GPU Parallel]
    end

    subgraph Sync["Sync Layer"]
        AS[AutoSyncDaemon]
        SR[SyncRouter]
        BW[SyncBandwidth]
    end

    subgraph Targets["Sync Targets"]
        TN1[Training Node 1]
        TN2[Training Node 2]
        TN3[Training Node 3]
    end

    SP -->|generates| DB[(Game DB)]
    GPU -->|generates| DB

    DB -->|DATA_SYNC_STARTED| AS
    AS --> SR
    SR --> BW

    BW -->|rsync| TN1
    BW -->|rsync| TN2
    BW -->|rsync| TN3

    TN1 -->|DATA_SYNC_COMPLETED| Pipeline[DataPipeline]
    TN2 -->|DATA_SYNC_COMPLETED| Pipeline
    TN3 -->|DATA_SYNC_COMPLETED| Pipeline
```

## Daemon Lifecycle

```mermaid
stateDiagram-v2
    [*] --> PENDING: create()
    PENDING --> STARTING: start()
    STARTING --> RUNNING: initialization complete
    STARTING --> FAILED: init error

    RUNNING --> STOPPING: stop()
    RUNNING --> FAILED: runtime error

    STOPPING --> STOPPED: cleanup complete
    FAILED --> STARTING: auto-restart

    STOPPED --> [*]
    FAILED --> PERMANENTLY_FAILED: max retries

    PERMANENTLY_FAILED --> [*]
```

## P2P Leader Election

```mermaid
sequenceDiagram
    participant N1 as Node 1
    participant N2 as Node 2
    participant N3 as Node 3 (Leader)
    participant V as Voter Quorum

    Note over N1,V: Normal Operation
    N3->>N1: Heartbeat
    N3->>N2: Heartbeat

    Note over N1,V: Leader Failure Detected
    N1->>N1: 6 consecutive probe failures
    N1->>V: Request Election

    V->>V: Verify quorum (4/7)
    V->>N1: Election approved

    N1->>N1: Become leader
    N1->>N2: LEADER_ELECTED event
    N1->>N3: LEADER_ELECTED event

    Note over N1,V: New Leader Active
    N1->>N2: Heartbeat
    N1->>N3: Heartbeat (when recovered)
```

## 48-Hour Autonomous Operation

```mermaid
flowchart TB
    subgraph Watchdogs["Recovery Watchdogs"]
        PW[ProgressWatchdog]
        P2PR[P2PRecoveryDaemon]
        SF[StaleFallback]
        MM[MemoryMonitor]
    end

    subgraph Detection["Problem Detection"]
        PW -->|PROGRESS_STALL_DETECTED| SS[SelfplayScheduler]
        P2PR -->|monitors| P2P[P2P Orchestrator]
        SF -->|sync failure| Models[Model Cache]
        MM -->|MEMORY_PRESSURE| Spawner[Job Spawner]
    end

    subgraph Recovery["Recovery Actions"]
        SS -->|priority boost| Priority[Stalled Config]
        P2PR -->|restart| P2P
        SF -->|use older model| Models
        MM -->|pause jobs| Spawner
    end

    subgraph Events["Recovery Events"]
        Priority -->|PROGRESS_RECOVERED| Metrics[Cluster Metrics]
        P2P -->|P2P_CLUSTER_HEALTHY| Metrics
        Models -->|MODEL_DISTRIBUTION_COMPLETE| Metrics
        Spawner -->|backpressure released| Metrics
    end
```

## Event Subscription Counts by Category

| Category   | Events | Key Events                                             |
| ---------- | ------ | ------------------------------------------------------ |
| Training   | 15     | TRAINING_COMPLETED, TRAINING_FAILED, TRAINING_PROGRESS |
| Selfplay   | 12     | SELFPLAY_COMPLETE, NEW_GAMES_AVAILABLE                 |
| Evaluation | 8      | EVALUATION_COMPLETED, EVALUATION_BACKPRESSURE          |
| Promotion  | 9      | MODEL_PROMOTED, PROMOTION_ROLLED_BACK                  |
| Sync       | 11     | DATA_SYNC_COMPLETED, P2P_MODEL_SYNCED                  |
| Health     | 26     | HOST_OFFLINE, P2P_CLUSTER_UNHEALTHY, LEADER_ELECTED    |
| Daemon     | 13     | DAEMON_STARTED, DAEMON_PERMANENTLY_FAILED              |
| Quality    | 10     | QUALITY_SCORE_UPDATED, QUALITY_DEGRADED                |
| Curriculum | 8      | CURRICULUM_REBALANCED, TIER_PROMOTION                  |
| Work Queue | 12     | WORK_CLAIMED, WORK_COMPLETED, WORK_FAILED              |

**Total Events**: 292 DataEventType members

## See Also

- [EVENT_FLOW_INTEGRATION.md](EVENT_FLOW_INTEGRATION.md) - Detailed event chain documentation
- [EVENT_SUBSCRIPTION_MATRIX.md](EVENT_SUBSCRIPTION_MATRIX.md) - Full emitter/subscriber matrix
- [FEEDBACK_LOOP_WIRING.md](FEEDBACK_LOOP_WIRING.md) - Feedback loop details
- [DAEMON_LIFECYCLE.md](DAEMON_LIFECYCLE.md) - Daemon state management
- `app/coordination/data_events.py` - Event type definitions
- `app/coordination/event_router.py` - Event bus implementation
