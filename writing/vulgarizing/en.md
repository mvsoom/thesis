# Vulgarizing abstract

<!--
Our speech conveys much more than just words. From a single pressure signal reaching the eardrum, we effortlessly perceive metadata such as intent, speaker identity, emotional state, approximate age and size, and even signs of illness. Behind all this lies a remarkably complex physiological process, controlled by the brain and driven by airflow from the lungs, rapid vocal fold motion, and the resonant filtering of the vocal tract.
-->

In the past decade, we have made enormous strides in recognizing and synthesizing speech using large "black-box" models trained on massive datasets. Today, these models have learnt to speak and understand major world languages such as English, Chinese, and Spanish with impressive accuracy. Remarkably, they remain largely agnostic to the physical mechanisms underlying speech, relying instead on large-scale pattern recognition and vast quantities of data.

In contrast, "white-box" models grounded in accumulated scientific knowledge do focus explicitly on these physical mechanisms. While they are not remotely competitive with black-box models in speech recognition or synthesis, they do offer the valuable ability to extract biologically meaningful information about the speaker from the signal itself.

This is important. A growing body of evidence shows that quantitative features of voice and speech can act as digital biomarkers of neurological and clinical conditions. In Parkinson's disease, for example, changes in speech and language have been reported to precede diagnosis by as much as a decade, and acoustic measures such as voice quality, articulation rate, and prosody have shown promise for early detection and monitoring. Similarly, alterations in specific acoustic parameters have been linked to mood disorders, suggesting that speech analysis could complement clinical assessment of depression.

Nevertheless, white-box models often lack the flexibility of black-box models needed to deal with real-world speech. I therefore propose a "grey-box" approach that attempts to combine the best of both worlds. By integrating physical insight with data-driven modeling, I derive a class of grey-box models that allows for far more accurate analysis and decomposition of speech into separate contributions from the vocal folds and the vocal tract. These models are also capable of indicating how certain they are of their measurements, while running close to real-time speed on modern hardware.

This was achieved by combining new theoretical insights in glottal flow and vocal tract models with customized machine learning models. I hope that this work can contribute to a more reliable and holistic use of speech analysis in clinical practice, forensic work, and the scientific study of vocalization as a whole.