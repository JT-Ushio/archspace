# 独立UG残差FFN任务记录

2026-10-09：准备L1及R0–R3，全部797,050,880参数。实际提交响应及状态以jobs.json和本文件后续记录为准。

Submitted L1: `mose-ffnres-l1-p1-1x8-1009-1601-8558-1220497`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Residual_Bottlenecks/run_l1.sh`.

Submitted R0: `mose-ffnres-r0-p1-1x8-1009-1601-2d86-37284085`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Residual_Bottlenecks/run_r0.sh`.

Submitted R1: `mose-ffnres-r1-p1-1x8-1009-1601-1eb9-88370018`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Residual_Bottlenecks/run_r1.sh`.

Submitted R2: `mose-ffnres-r2-p1-1x8-1009-1601-3788-1295629`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Residual_Bottlenecks/run_r2.sh`.

Submitted R3: `mose-ffnres-r3-p1-1x8-1009-1601-c6f0-34872256`, pool 1x8, priority1; script `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/shijiayang-p/archspace-MoSE/reproduce/olmo-core-backend/scripts/train/FFN_Residual_Bottlenecks/run_r3.sh`.

## 2026-10-09 16:04 HKT — monitoring

- l1: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r0: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r1: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r2: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r3: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=981; {'step': 980, 'ce': 3.229}.

五条均已确认创建；首轮16:04 HKT全部Inqueue、active=0。原C9为Running、step981，最近step980 CE3.229。已启动本地1800秒监控循环，包含本组五条和C9；不因排队重复提交。HF meta结构检查通过，原生8项行为检查通过，HF数值验证not_run。

## 2026-10-09 16:12 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r0: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r1: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r2: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r3: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1108; {'step': 1100, 'ce': 3.104}.

## 2026-10-09 16:45 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=593; {'step': 590, 'ce': 3.351}.
- r0: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r1: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r2: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- r3: Inqueue; {'active': 0, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=None; None.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1767; {'step': 1760, 'ce': 2.965}.

## 2026-10-09 17:15 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1177; {'step': 1170, 'ce': 3.101}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=484; {'step': 480, 'ce': 3.463}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=349; {'step': 340, 'ce': 3.956}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=372; {'step': 370, 'ce': 3.772}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=201; {'step': 200, 'ce': 5.193}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2352; {'step': 2350, 'ce': 2.88}.

## 2026-10-09 17:15 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1183; {'step': 1180, 'ce': 3.054}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=490; {'step': 490, 'ce': 3.523}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=354; {'step': 350, 'ce': 3.844}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=378; {'step': 370, 'ce': 3.772}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=207; {'step': 200, 'ce': 5.193}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2357; {'step': 2350, 'ce': 2.88}.

## 2026-10-09 17:45 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1776; {'step': 1770, 'ce': 2.95}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1116; {'step': 1110, 'ce': 3.13}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=870; {'step': 870, 'ce': 3.366}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=999; {'step': 990, 'ce': 3.246}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=843; {'step': 840, 'ce': 3.234}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2947; {'step': 2940, 'ce': 2.837}.

## 2026-10-09 18:16 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2395; {'step': 2390, 'ce': 2.931}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1796; {'step': 1790, 'ce': 2.982}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1410; {'step': 1410, 'ce': 3.15}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1665; {'step': 1660, 'ce': 3.045}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1499; {'step': 1490, 'ce': 3.094}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3537; {'step': 3530, 'ce': 2.761}.

## 2026-10-09 18:45 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2967; {'step': 2960, 'ce': 2.88}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2426; {'step': 2420, 'ce': 2.948}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=1907; {'step': 1900, 'ce': 3.099}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2284; {'step': 2280, 'ce': 2.912}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2099; {'step': 2090, 'ce': 3.074}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4106; {'step': 4100, 'ce': 2.751}.

## 2026-10-09 19:15 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3538; {'step': 3530, 'ce': 2.762}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3059; {'step': 3050, 'ce': 2.911}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2426; {'step': 2420, 'ce': 3.013}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2929; {'step': 2920, 'ce': 2.774}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2740; {'step': 2740, 'ce': 2.983}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4695; {'step': 4690, 'ce': 2.699}.

## 2026-10-09 19:45 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4134; {'step': 4130, 'ce': 2.76}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3712; {'step': 3710, 'ce': 2.84}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=2945; {'step': 2940, 'ce': 2.95}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3550; {'step': 3550, 'ce': 2.856}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3381; {'step': 3380, 'ce': 2.921}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5286; {'step': 5280, 'ce': 2.707}.

## 2026-10-09 20:09 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4598; {'step': 4590, 'ce': 2.666}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4231; {'step': 4230, 'ce': 2.837}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3354; {'step': 3350, 'ce': 2.93}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4059; {'step': 4050, 'ce': 2.793}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3887; {'step': 3880, 'ce': 2.943}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5754; {'step': 5750, 'ce': 2.638}.

## 2026-10-09 20:15 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4722; {'step': 4720, 'ce': 2.656}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4368; {'step': 4360, 'ce': 2.713}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3464; {'step': 3460, 'ce': 2.958}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4194; {'step': 4190, 'ce': 2.756}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3999; {'step': 3990, 'ce': 2.871}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5877; {'step': 5870, 'ce': 2.712}.

## 2026-10-09 20:45 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5314; {'step': 5310, 'ce': 2.703}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4999; {'step': 4990, 'ce': 2.794}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=3983; {'step': 3980, 'ce': 2.949}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4831; {'step': 4830, 'ce': 2.748}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4636; {'step': 4630, 'ce': 2.762}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=6470; {'step': 6470, 'ce': 2.707}.

## 2026-10-09 21:15 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5903; {'step': 5900, 'ce': 2.686}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5649; {'step': 5640, 'ce': 2.694}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4498; {'step': 4490, 'ce': 2.881}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5471; {'step': 5470, 'ce': 2.716}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5274; {'step': 5270, 'ce': 2.841}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=7030; {'step': 7030, 'ce': 2.598}.

## 2026-10-09 21:45 HKT — monitoring

- l1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=6499; {'step': 6490, 'ce': 2.621}.
- r0: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=6302; {'step': 6300, 'ce': 2.766}.
- r1: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=4999; {'step': 4990, 'ce': 2.862}.
- r2: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=6085; {'step': 6080, 'ce': 2.704}.
- r3: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=5915; {'step': 5910, 'ce': 2.78}.
- c9: Running; {'active': 1, 'succeeded': 0, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=7616; {'step': 7610, 'ce': 2.614}.

## 2026-10-10 22:02 HKT — monitoring

- l1: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=14306; {'step': 14300, 'ce': 2.505}.
- r0: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=14306; {'step': 14300, 'ce': 2.555}.
- r1: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=14306; {'step': 14300, 'ce': 2.63}.
- r2: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=14306; {'step': 14300, 'ce': 2.555}.
- r3: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=14306; {'step': 14300, 'ce': 2.631}.
- c9: Succeeded; {'active': 0, 'succeeded': 1, 'failed': 0, 'created_failed': 0, 'pending': 0, 'stopped': 0}; step=14306; {'step': 14300, 'ce': 2.504}.
