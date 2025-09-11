# 🧠 Project: Adaptive Rehearsal in Continual Learning using Concept Drift Detection

![Status](https://img.shields.io/badge/status-In%20Progress-yellow)

A Major Project by **[Your Name]**

---

## 🎯 1. Project Abstract

This project addresses the challenge of **catastrophic forgetting** in continual learning. Standard rehearsal-based methods, while effective, are computationally inefficient due to their static, "always-on" replay strategy. This project proposes a novel **Smart Rehearsal** system that integrates a state-of-the-art rehearsal method (iCaRL) with a formal concept drift detector (ADWIN).  

The core hypothesis is that by triggering intensive rehearsal *only when the model shows statistical signs of forgetting*, we can achieve comparable accuracy to existing methods but with **significantly lower computational cost**, paving the way for more efficient lifelong learning systems.

---

## 💡 2. The Research Journey & Motivation

The central problem in continual learning is that neural networks tend to forget old knowledge when learning new tasks. My initial research, documented in `progress_report.html`, involved a systematic exploration of existing solutions:

1. **Surveying the Field**  
   Understood the three primary families of solutions: Regularization, Architectural, and Rehearsal methods.
2. **Evaluating Strategies**  
   Based on foundational papers like van de Ven & Tolias (2019), it became clear that Rehearsal methods offer the most robust and effective baseline for challenging Class-Incremental scenarios.
3. **Identifying a Gap**  
   While state-of-the-art rehearsal methods like iCaRL (Rebuffi et al., 2017) are powerful, they are inefficient. They replay memories at a constant rate, regardless of the model's actual performance.
4. **Formulating a Hypothesis**  
   The key insight was to re-frame "forgetting" as a detectable event. By borrowing the concept of **Concept Drift Detection** from the stream mining community, we can create a system that intelligently decides *when* to rehearse.

>
