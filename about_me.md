;; ============================================================================
;; ABOUT ME — edit this file freely. It is rendered by the "About Me" page.
;;
;; The few KEY: value lines below (up to the first --- line) are settings the
;; page reads. Everything after the first --- is your bio, written in plain
;; Markdown. Lines starting with ;; are comments and are ignored.
;;
;;   PHOTO    : path to your photo, relative to the project root.
;;              Drop the image in the project (e.g. assets/about_me.jpg) and put
;;              its path here. If the file is missing, a placeholder is shown.
;;   LINKEDIN : your LinkedIn profile URL.
;;   EMAIL    : your contact email address.
;;   GITHUB   : your GitHub profile URL.
;; ============================================================================

PHOTO: assets/about_me.jpg
LINKEDIN: https://www.linkedin.com/in/samratr/
EMAIL: rsamrat@dontsp.am
GITHUB: https://github.com/reamgripper/wc2026-predictor
---



 I am a supply chain professional and a self-proclaimed math geek. No, That doesn't mean I’m not a human computer who experiences numbers as colors (<a href="https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(00)01571-0?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1364661300015710%3Fshowall%3Dtrue" target="_blank">Colour–Text Synaesthesia</a>). I’m just someone deeply passionate about the elegance of numbers—from high school algebra and classic algorithms to Hadamard gates and Tensor Products in Quantum computing.

 Modern mathematics has moved beyond pen and paper and almost all of the mathematicians that I have know are proficient in writing codes.
For a long time, there was just one catch: I used to hate writing code from scratch. I could write the basic ones, but the moment it became complex, my codes returned nothing but errors.

That hurdle completely vanished with the advent of Large Language Models. Now, before my software engineering friends get defensive, let me clarify: I am not suggesting I’m building full-stack applications or managing secure, stable enterprise deployments. That remains firmly in the domain of engineering experts. But we all have to accept the reality that "Intent driven Coding" is the future. I highly recommend watching this keynote by Andrew Ng on 
<a href="https://www.youtube.com/watch?v=g8um2AEf5ZA" target="_blank">the future of software engineering</a>.

What LLMs do allow me to do is bridge the gap between theory and execution. They handle the syntax so I can focus on the strategy - accelerating how I write code to build complex mathematical models for real-world decision-making. And that is why i decided to embark on a weekend project right before the start of World Cup 2026.
Before I explain my weekend project, let me tell you an ancient indian fable which talks about the perils of believing in crowd wisdom. A priest was walking home, carrying a healthy goat he had received as a reward for his services. Along his journey, he encountered three tricksters stationed at different intervals along the road, each determined to steal the animal without using force.

Instead of attacking, they chose to manipulate his reality.

The first trickster stopped him, asking why a holy man would carry a filthy dog on his shoulders. The priest scoffed and marched on. Soon after, the second trickster approached, expressing shock that he was carrying a dead calf. Doubt began to creep in. Finally, the third trickster laughed and asked why he was lugging around a donkey.

Convinced by the sheer repetition that three independent sources could not possibly be wrong, the priest decided his own senses were lying to him. Fearing the animal was actually a shapeshifting demon, he panicked, threw it to the ground, and fled—leaving the tricksters to quietly enjoy the prize they had won through pure psychological manipulation. 
Most people believe crowd wisdom is flawed because it is full of biases and manipulation. But something has fundamentally changed due to anonymous betting markets. 

Over the weekend in early June of 2026, I spent some time bulding a prediction system for Football matches. Now this is the most clichéd thing almost every data scientists has work on, but I wanted to try a different hypothesis. I decieded to blend Betting market biases into prediction as well. Given that someone out there by the name of 
<a href="https://www.wsj.com/finance/how-the-trump-whale-correctly-called-the-election-cb7eef1d" target="_blank">
Théo successfully predicted US election results 
,</a> and the market collectively is 
<a href="https://www.cityam.com/how-are-prediction-markets-like-polymarket-more-accurate-than-wall-street-analysts/" target="_blank">
predicting Earnings better than wallstreet analysts,</a>.

we have to start paying attention to crowdsourced knowledge. Perhaps it has to do with the fact that due to anonymity, someone can use insider knowledge, Or it has to do with the fact that Collective intelligence of the crowd is better than the individual intelligence.

So i decided to use a simple predictor using Poisson GLM(Generalised Linear Model) and combine with Elo ratings(named after Arpad Elo) to predict the probability distribution of Lambda(λ). 
I started using this to predict matches for a fantasy football that my neighborhood is running and I found the model quite conservative.
So i decided to look up the bets on Draftkings and Polymarket. I noticed that the predictions were fairly accurate for the major events of Soccer. Perhpas die hard soccer fans know how well the players will play. Or perhaps something else. 
I decided to blend data from both the models into the prediction model and Viola!! 
So far the model is holding well but i will keep you posted on how this evolves. 
Let us not forget, no mathematical model can predict human will, a key determinant in outcome of matches.

To be continued...

