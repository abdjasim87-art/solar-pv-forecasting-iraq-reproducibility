PATCHTST LOCO HISTORY-AWARE ZERO-SHOT v1 — START HERE

هذه هي التجربة الأخيرة المقترحة قبل إغلاق ملف التجارب.

مهم علمياً:
هذا PatchTST النهائي يعمل S-mode بمدخل واحد هو تاريخ PV الماضي، لذلك تجربة LOCO هنا
history-aware zero-shot وليست pure cold-start. المدينة المستهدفة لا تدخل إطلاقاً في Train/Validation/scaler/correction،
لكن تاريخها الماضي المتاح عند وقت التنبؤ يدخل كمدخل كما في PatchTST النهائي.

المصفوفة المجمدة:
15 مدينة محجوبة × H24/H48 × seeds 1,2,3 = 90 تدريباً.
Lookback ثابت L168 في H24 وH48، موروث من نتائج PatchTST النهائية المدققة. لا يوجد LOCO tuning للـlookback.

بعد فك الحزمة على RunPod:
  cd /workspace/lstm
  tar -xzf patchtst_loco_history_zero_shot_v1.tar.gz
  chmod +x patchtst_loco_history_zero_shot_v1/*.sh

شغّل فقط preflight أولاً:
  ./patchtst_loco_history_zero_shot_v1/00_preflight.sh

ثم توقف وارفع:
  /workspace/lstm/patchtst_loco_v1_00_preflight.log
  /workspace/lstm/patchtst_loco_v1_00_preflight_audit.tar.gz

لا تشغّل 10 أو 20 أو أي Test قبل مراجعة preflight.
