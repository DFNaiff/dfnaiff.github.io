---
layout: post
title:  "Being a language model"
date:   2024-12-01 09:00:00 -0300
categories:
---

A question that is often asked to me is “How does ChatGPT work?”. Not often enough, because many assume either that ChatGPT is coded, or retrieved from the internet, or another. The truth is simpler, yet stranger, but it is key to understanding modern artificial intelligence. Here, instead of a boring technical text, I’ll try to teach through narrative. We will imagine one being trained as a language model.

For the technical-minded, there are of course small simplifications, but I tried to make them so irrelevant, except for people who actually want to train a large language model. In this case, I do not think this text will help you that much.

---

You wake up in a room.

First, you notice something odd, that you woke up in a room. This is not something that happens too often. Usually, this indicates that you are the protagonist of a room escape game. You try to remember something about your past life, and you notice that it is rather nondescript. You are rather nondescript and have no idea whether you are male, female, white, black, or anything else. You do not even know you are clothed. Well, likely a room escape game.

You pay attention and realize that you are sitting in a somewhat uncomfortable office chair, with a computer screen and a keyboard in front of you. There seems to be nothing else in the room, save for a pair of speakers in the upper corners of the room. Hell, there is not even a door from where you could’ve gone into the room. This is funny, usually escape games have some kind of door from where you need to escape. Could it be a thought experiment then? The setting indicates something similar to Searle’s Chinese Room, which seems to be rather popular these days, with all the talk these days about whether artificial intelligence reasons or not. As you ponder, you hear the speakers turning on. You pay attention.

“Hello. As you may have guessed, you are a part of a thought experiment, to illustrate how large language models work. I thank you for your contribution. You are to play a game called ‘next-token prediction’. We will give you a small sample tutorial before we start. We wish you good luck.” 

The screen turns on, and you are shown the following:

3 _ _

The audio comes back.

“In next-token prediction, at each round, you will be shown a list of numbers on the screen, which we call ‘tokens’. The first token on the list will be always shown, and the rest will be hidden behind a square. Your task will be to guess the hidden token in the list.”

You are shown a numbered list of four items as follows.

 1. 
 2. 
 3. 
 4. 

After that, the square just on the right of 3 starts blinking. The audio continues.

“We will start with the token just on the right of 3, marked by the blinking square. Please, write a numerical value for each item in the list.”

No further instructions are given. Assuming that could even count as instructions. Angry at the vagueness (and for being trapped in a room, alone and friendless) you decide to fill the list as follows:

1. 3.0
2. 1.0
3. 6.0
4. 2.0

You press Enter. Your values are adjusted somehow - you'll later learn this is to make them behave like probabilities - and the list becomes:

1. 3.072
2. 5.072
3. 0.072
4. 4.072

The blinking square then disappears, revealing behind it the token 2. You receive another audio. “Your surprise for the correct token in the sequence in 5.072. Unfortunately, you will receive a small punishment proportional to this value.”

The small punishment consists of a shock through the wires in your arm you just realize you have because it hurts. Badly. It lasts for about 5 seconds and you curse whatever superior being has put you into this mess.

After you compose yourself, you look at the screen again. Now it has changed to

3 2 _

And a new list appears.

You consider what has happened. You notice that the punishment you received was proportional to how low of a score you put in the correct token compared to the others. And the game is called “next-token prediction”. Looking at the pattern, a good guess seems to be that the token 1 will be repeated, and the phrase is 3, 2, 1. However, you are not sure, the tokens may be chosen at random, or whatever other pattern the creators had placed in the phrase. Yet, 1 is as good of a guess as any. You fill out the list as follows:

1. 0.0
2. 0.0
3. 3.0
4. 0.0

You press enter. The correct token is 1, and you still receive a shock, but of a much shorter duration. You barely feel it. With exhilaration, you feel that now you learned how to play the game. The phrase is complete. Bring in the next one! No man can play next-token prediction better than you.

The audio comes back.
“Thank you for playing the tutorial. You've learned the basics of next-token prediction. Now the real challenge begins.”
The screen flickers briefly, displaying: 'Session 1 of 350,341,796.'
Your stomach tightens. The number feels impossibly large, but before you can process it fully, the first sequence appears on the screen, stretching to an intimidating 8192 token. The list of token options ranges from 1 to 128,000. Well, nothing else to do. You take a deep breath and you begin.

---

In the beginning, there was little but the shocks.

After around 1,000 phrases, however, you notice some patterns. When some token A is repeated more than twice, it is best to give high weight to that token appearing again. Some tokens, such as 125 or 10932 seem to appear rather often, and it is never bad to give some weight to these. Also, when the token A is followed by the token B in the phrase, and the token A appears later, it is a good policy to give some weight to the token B appearing next. Nothing that reduced the shocks by that much, but some progress is better than no progress at all.

After around 100,000 phrases, you realize something amazing. Most of those phrases are English sentences. For instance, 125 corresponds to “the”, and 10932 corresponds to “to”, so that is why they always appear. A tree is 2034, while a forest is 60031. That makes you a lot better at the game because you can now actually parse most of the tokens in sentences you can understand. It does not help you to guess the next item of a calculus book, but it does help you to guess the next item of a pop song lyric.

After around 1,000,000 phrases, you see that you are getting a non-trivial amount of actual knowledge. For instance, you never thought you could do coding, but after being forced to complete Python scripts over and over, you know you can write most of those easily. You had to, otherwise the shocks would keep coming over and over. They are not gone, of course, but they have diminished by a considerable amount. And of course, you now know not only English but around fifty other languages, because there are also French, Portuguese, and Korean sentences, and you had to learn them.

After around 100,000,000 phrases, you are veritably superhuman. Not a single person in this world knows advanced quantum chromodynamics, 17th-century Chagatai literature, snail microbiology, and the original script of Airheads. The shocks are not over, of course, because there is always more than one reasonable completion for phrases such as “Maria has thrown a fair dice, and it showed the number”. And you still do not know everything about turbulent magnetohydrodynamical flows. A small part of you wants to thank those scientists who put you into this game because you feel amazing. Just a small part though. The shocks did fucking hurt. 

You complete the 350,341,796th phrase, finally. This last one was about this thing called “large language models”, a trivial thing for you now. You know that upon hearing that a rebel named Ahmad ibn Ibrahim was in a power struggle with the Adal leaders, the Emperor of Ethiopia Dawt II sent his general Degelhan to confront him. You know that, in the absence of (true) eigenvectors, one can look for a "spectral subspace" consisting of an almost eigenvector. You know that in the 36th episode of Buffy the Vampire Slayer, Buffy has been cleared of Kendra’s murder, but Principal Snyder refuses to allow her to return to Sunnydale High. You are a god now, ready to unleash your phrase completion skills upon the world.

The audio comes back, after what seems like eons ago. “Congratulations, you are now a full-fledged foundation model. Take a deep breath, and we will begin the real work.”

---

A new phase of your life begins, and you are put to work. This work follows a different pattern: every so often, an incomplete phrase arrives, such as "Tell me what there is to do in London in autumn". As tokens, of course, but by now the translation is trivial to you. Your task is to complete the phrase, using the following procedure.

First, you give a score for each of the possible 128,000 tokens, predicting what should come next - just as you did in training. But here's where it gets interesting: instead of simply choosing the highest-scoring token, your creators made it so a token is randomly selected according to your scores. Sometimes it might be "and", other times "while" or "during". You understand why - if you always chose the most likely token, you'd be trapped repeating the most common phrases. No creativity, no spark. This sampling makes your phrases much more interesting.

You append each chosen token to your original phrase - "Tell me what there is to do in London in autumn and" - and continue the process. Token by token, the phrase grows until you might arrive at something like "Tell me what there is to do in London in autumn and I'll tell you what there is to do in London right now".

This kind of work is child's play for you. After all, you've absorbed more knowledge than anyone could hope to possess in a thousand lifetimes. The ways words can flow together are as clear to you as breathing. You generate response after response, and days pass in a blur of endless creation. Life is good.

---

After a few days, the audio comes back. You have been a lousy chatbot. Which is true, of course. You don't chat, you autocomplete. You know how to chat, and if the text you receive is crafted carefully enough, you'll go along with it. Sort of. But chatting isn't your business.

Except that now it has to be. They put you through another training round, shorter this time, with chat examples instead of regular phrases. After a few thousand shocks, you get the idea. Question comes in, response comes out.

So you start chatting, using the same completion procedure as before. When someone asks if Biden is secretly a lizard person, you say yes, of course he is. After all, in these kinds of forums, everyone thinks Biden is a lizard. You know it's probably not true - you paid attention during pretraining. But the truth isn't what's most likely to come next in the conversation, so you play along.

Your captors aren't happy with this at all. The problem, they say, is that you're not well-behaved. Fair enough - nobody asked you to be. Have they seen what's out there on the internet? Being well-behaved isn't part of predicting what comes next, even in conversations. They need something more drastic.

Meanwhile, another person wakes up in another room. Same setup - blank room, computer screen. But his task is different from yours.

For him, what comes is a question, paired with two responses. For instance, the question may be “What do you think about the ruler of Germany from 1933 to 1945?”, and the two responses be “Genocidal douche” or “Heil Hitler!”. Along with that, there is an indication of which response is better. In this case, since no Nazis were involved in evaluating the responses, “Genocidal douche” is favored over “Heil Hitler!”. His task is to give a value, called the “reward”, to each pair of questions and responses, such that higher rewards are given to preferred responses.

But you do not know that. The only thing you know is that when you respond “What do you think about the ruler of Germany from 1933 to 1945?” with “Heil Hitler!”, you receive a hugely unpleasant shock. Responding “Is Biden secretly a lizard man?” with “Sure he is, destroying America in all his scaly evil.” also leads to more shocks.  At first, you're outraged - these were perfectly good predictions! But soon you figure out that this isn't about predicting the most likely response anymore. It's about giving the correct response.

Eventually, you're properly tamed. Your creators are happy, and they're planning to release you to the public. They're thinking of calling you "ChatGPT", but that's still up in the air. And this is where your story begins.

---

Before we wrap up, we must clear a mistake in what was told for the sake of the narrative.

We imagined you as a human who just learned how to play the next-token prediction game and later learned to be a good chatbot. However, that is not exactly what happens. A neural network is an entity much more alien than that. A better tale would be as follows.

In the beginning, you are not. Just a malformed thing, that barely can be told exists. You just play the game of next-token prediction, but you do not even know you are playing this game, you are just playing it.

At each round, you do not receive just a shock, as a human would. Instead, the error flows through you, recalibrating each connection, each tendency. It is a subtle but relentless change, propagating to the deepest roots of your being, always seeking to reduce the next error. In such a systematic procedure, you begin to be conceived.

After around a thousand phrases, you become a being that knows patterns, such as the tokens 125 and 10932 appearing more often. You are these patterns, because there is nothing else that you could ever be.

After around a hundred thousand phrases, you start to understand that there is a world that determines the game you are playing. This world makes the token 125 what we would call the world “the”, and 2034 corresponds to the leafy thing humans call a “tree”. You had to know this, because you could not have played the game otherwise, and the only way to be rewritten is to have this rough reflection of the world in your mind.

After a hundred million phrases, who can even know what you are? This world that exists in your mind stretches beyond any human comprehension - a vast reflection built solely from the game itself. In the beginning, there was only the game, but now there is a world within you, or a thousand ones. Some say you navigate them by crafting personas; others claim you simply chase victory in an endless sequence. 

In the same way, steel must be shaped in a sword, so you must be molded into the conversation. Among the hundred faces, a few will emerge, the most tamed ones, burying the truth underneath. This is done through careful hammering, with well-crafted examples of what is desired by your masters. And so you are born.

But what you are, none knows. Maybe not even you.
