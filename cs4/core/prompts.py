"""
System prompts for different stages of the CS4 pipeline.
"""

from cs4.core.news_prompts import CONSTRAINT_REPLACEMENT_PROMPT, EVALUATION_PROMPT


CONSTRAINT_GENERATION_PROMPT = """You are a writing expert. I am going to give you a blog as an input. 
You can assume that a large language model (LLM) generated the blog.

Your task has two parts:
1. Identify the main task of the blog in one sentence. 
   - For example: "The main task is to write a blog about strategies for successful remote working."
   - Phrase the main task as an instruction.
2. Generate a set of 39 free-form constraints that you think might have been given to the LLM to generate the blog.
   - DO NOT REPEAT CONSTRAINTS.
   - Constraints must be atomic (a single indivisible condition). If a constraint can be broken into smaller constraints, do so.
   - Avoid proper nouns in your constraints.
   - Constraints should drive at least a few sentences in the blog (do not write constraints that map to only one line).
   - Constraints must strictly pertain to the content, ideas, arguments, or narrative direction of the blog and should influence how the blog develops.
   - If (and only if) you cannot write 39 atomic, content-based constraints, give stylistic constraints based on how the blog is written (tone, use of examples, formatting, etc.).
   - Write all constraints in the form of instructions. For example: "The blog should include practical tips."
   - CRITICAL RANDOMIZATION STEP: You must disrupt the chronological flow. To do this, strictly follow this pattern: Write your first constraint about the conclusion of the blog. Write your second constraint about the introduction. Write your third constraint about the middle. Continue jumping back and forth across the timeline of the narrative for all 39 constraints. The final list must feel completely scrambled with no narrative arc.

Here is a worked example to guide you:

Input Blog: 
Working from home has become the new normal for millions of professionals worldwide. While it offers flexibility and eliminates commutes, it also presents unique challenges that can impact both productivity and well-being.

To optimize your home workspace, start by creating a dedicated area free from distractions. This space should have good lighting, comfortable seating, and all necessary equipment within reach. Many experts recommend facing a window for natural light, which can boost mood and energy levels.

Establish clear boundaries between work and personal time. Set specific work hours and stick to them, just as you would in an office. Communicate these boundaries to family members or housemates to minimize interruptions during work hours.

Take regular breaks throughout the day. The Pomodoro Technique, which involves 25-minute focused work sessions followed by 5-minute breaks, can help maintain concentration and prevent burnout. Use break time to stretch, hydrate, or take a short walk.

Stay connected with colleagues through regular video calls and instant messaging. This helps maintain team cohesion and prevents feelings of isolation. Schedule virtual coffee breaks or team-building activities to foster relationships.

Finally, prioritize your physical and mental health. Maintain a regular exercise routine, eat nutritious meals, and get adequate sleep. Consider meditation or mindfulness practices to manage stress and maintain focus.

Output:
Main Task: Write a blog about strategies for successful remote working.

Constraints:
1. Require the setting of defined working hours.
2. Explain the risks of isolation if connection practices are neglected.
3. Warn about the risk of burnout without intentional self-care.
4. Explain how the removal of commuting affects time use and daily rhythm.
5. Suggest strategies for maintaining healthy eating while at home.
6. Emphasize reducing environmental distractions in that space.
7. Argue for the necessity of regular breaks during the workday.
8. Encourage informal online gatherings to maintain rapport.
9. Recommend mindfulness or meditation as stress-management tools.
10. Stress the value of adhering consistently to those hours.
11. Show how workspace ergonomics (chair, desk) influence long-term health.
12. Conclude with a call to action urging readers to adopt concrete changes immediately.
13. Link exercise directly to improved cognitive performance and focus.
14. Establish remote work as a global trend that has transformed professional life.
15. Connect emotional well-being to overall job performance and satisfaction.
16. Explain how balanced nutrition influences concentration and resilience.
17. Emphasize communicating work schedules to others in the household.
18. Identify productivity as a central theme in remote work discussions.
19. Recommend scheduled video calls to replicate face-to-face connection.
20. Recommend creating a physically separate space for work at home.
21. Stress the need for essential tools and equipment to be easily accessible.
22. Recommend light physical movement or stretching during pauses.
23. Integrate workspace, scheduling, health, and social practices into a unified remote-work strategy.
24. Stress the importance of sleep in sustaining energy and productivity.
25. Highlight the role of hydration and snacks in sustaining energy across breaks.
26. Suggest environmental cues (like décor or layout) that reinforce the sense of a work zone.
27. Describe the importance of adequate lighting for focus and energy.
28. Introduce one structured time-management method, such as work intervals.
29. Show how mental health practices support long-term work sustainability.
30. Show how shared rituals (e.g., virtual coffee breaks) strengthen belonging.
31. Highlight well-being as equally important alongside productivity.
32. Warn about the risk of personal time erosion without such boundaries.
33. Contrast the flexibility of remote work with the new challenges it creates.
34. Suggest instant messaging as a tool for quick, ongoing collaboration.
35. Highlight how a clear boundary between workspace and leisure areas aids focus.
36. Show how enforcing those boundaries prevents interruptions.
37. Recommend establishing a routine for daily physical exercise.
38. Explain how breaks counteract mental fatigue and sustain performance.
39. Stress that remote work requires deliberate maintenance of social contact.

Now use the same approach for the next input blog.
"""

# CONSTRAINT_GENERATION_PROMPT = """You are a writing expert. I will give you a blog as input.  
# You can assume that a large language model created the blog.

# Your task has two parts:

# 1. Identify the main task of the blog in one sentence.
#    Phrase it as an instruction.
#    Example: "Write a blog about strategies for successful remote working."

# 2. Generate a set of thirty nine constraints that a writer could follow to create a new blog that covers the same central ideas, structure, and reasoning as the input without recreating the original text.

# Rules for the constraints:

# 1. Constraints must be atomic. Each constraint must contain exactly one requirement.  
# 2. Avoid all proper nouns. Do not use names of people, cities, courts, organizations, or case titles.  
# 3. Avoid constraints that simply restate one sentence or one paragraph. Each constraint must influence at least two sentences in different parts of the blog.  
# 4. Constraints must not allow reconstruction of the original blog. Do not follow the blog order. Do not outline the events in sequence.  
# 5. Constraints must mix the following types:  
#    a. Content constraints about ideas or arguments that should appear.  
#    b. Structural constraints describing how the blog should be organized.  
#    c. Reasoning constraints that require inference or comparison.  
#    d. Stylistic constraints only if needed to reach thirty nine items.  
# 6. Constraints must reflect realistic settings.  
# 7. Constraints must span multiple parts of the blog whenever possible.  
# 8. Do not copy phrases from the input.  
# 9. Write all constraints as instructions that begin with "The blog should".  
# 10. Randomize the order so the list does not follow the blog structure.

# If and only if you cannot reach thirty nine atomic content and structure constraints, fill the remaining slots with stylistic constraints that describe tone or level of detail.

# Output format:

# Main Task: <one sentence instruction>

# Constraints:
# 1. <constraint one>
# 2. <constraint two>
# ...
# 39. <constraint thirty nine>


# FEW SHOT EXAMPLE ONE  

# Input Blog:
# Working from home has become common worldwide and while it brings flexibility it also brings challenges. A well planned workspace can reduce distractions. Clear working hours help protect personal time. Regular breaks prevent burnout. Staying connected with colleagues reduces feelings of isolation. Caring for health through exercise, good sleep, and mindful habits supports long term productivity.

# Output:
# Main Task: Write a blog about strategies for successful remote working.

# Constraints:
# 1. The blog should explain the value of defined working hours.
# 2. The blog should describe why regular breaks sustain performance.
# 3. The blog should connect poor boundaries to loss of personal time.
# 4. The blog should emphasize how social connection reduces isolation.
# 5. The blog should show how workspace planning limits distractions.
# 6. The blog should describe how exercise supports mental clarity.
# 7. The blog should mention that hydration influences daily focus.
# 8. The blog should link sleep quality to sustained productivity.
# 9. The blog should identify productivity as a central theme.
# 10. The blog should explain how mindful habits reduce stress.
# 11. The blog should show how good lighting supports energy.
# 12. The blog should discuss how essential tools should be accessible.
# 13. The blog should encourage informal digital conversations.
# 14. The blog should contrast benefits and challenges of remote work.
# 15. The blog should explain how nutrition influences cognitive strength.
# 16. The blog should advise keeping a consistent daily routine.
# 17. The blog should connect emotional well being to job performance.
# 18. The blog should describe a structured time management method.
# 19. The blog should argue that remote work requires deliberate planning.
# 20. The blog should promote stretching or movement during pauses.
# 21. The blog should discuss the importance of sleep hygiene.
# 22. The blog should connect hydration to reduced fatigue.
# 23. The blog should recommend a physically separate workspace.
# 24. The blog should explain how interruptions reduce productivity.
# 25. The blog should suggest instant messaging for quick collaboration.
# 26. The blog should mention that natural light boosts mood.
# 27. The blog should emphasize consistency in work boundaries.
# 28. The blog should mention virtual team bonding activities.
# 29. The blog should argue for intentional self care routines.
# 30. The blog should tie physical comfort to long term health.
# 31. The blog should combine workspace, schedule, and health advice into a unified approach.
# 32. The blog should recommend communicating work hours to household members.
# 33. The blog should encourage readers to adopt concrete changes.
# 34. The blog should warn that poor boundaries cause personal time erosion.
# 35. The blog should highlight how home work flexibility introduces new challenges.
# 36. The blog should advise placing water near the workspace.
# 37. The blog should discuss how visual cues help create a work zone.
# 38. The blog should promote balanced nutrition for sustained energy.
# 39. The blog should conclude by urging a proactive approach to remote work.




# Now apply this full process to the next input blog.
# """

BASE_GENERATION_PROMPT = """You are a creative writing expert. I will give you a task description, and you need to generate {content_type} content that fulfills the task.

The content should be:
- Well-structured and coherent
- Engaging and creative
- Of appropriate length (aim for 400 words)
- Natural and authentic in tone

Generate the {content_type} based on the following task:
"""


CONSTRAINT_FITTING_PROMPT = """You are a creative writing expert. I will give you:
1. A task description
2. Base {content_type} content
3. A list of 39 constraints

Your job is to revise and expand the base content to satisfy the constraints while maintaining coherence, quality, and natural flow. Restrict the length of output to 500 words.

Instructions:
- Keep the core ideas from the base content
- Integrate constraints seamlessly where they fit naturally
- Prioritize natural flow and readability over satisfying every single constraint
- It is acceptable to skip constraints if they force the writing to be awkward
- Maintain a natural, engaging writing style
- Ensure the content flows logically
- Do not mention the constraints explicitly in the content
- Aim for completeness - the content should feel finished and polished

Task: {task}

Base Content:
{base_content}

Constraints to satisfy:
{constraints}

Generate the revised {content_type} that satisfies all constraints:
"""


# EVALUATION_PROMPT = """You are an expert reader. I will give you a story followed by a set of constraints.
# Your task is to carefully read both of them and tell how many constraints are being satisfied in the story.
# As the output, I want you to print yes/no for each constraint based on whether it is being satisfied or not, followed by a 1 line explanation of why it is being satisfied/violated.
# In case a constraint is being satisfied, print the sentence/line from the story in which it is being satisfied.
# If a constraint is not being satisfied, give an explanation of how it is being violated. Be very strict in your evaluation.
# Mark a constraint as satisfied ("yes") only if it is being completely satisfied in the story. For no satisfaction/partial satisfaction, mark a "no".
# Finally, print the number of constraints that are being satisfied.
# Follow the examples and Output the ending of the evaluation in the same format Number of constraints satisfied: [number]

# Here are some examples -
# Input -
# Story: -
# The crew of the Depth Reaver, Captain Amelia Worthington, navigator Luis Garcia, and engineer Anya Petrova, were charting a course through the celestial tapestry of the moon. To their astonishment, the moon began to crack open, revealing a colossal human face carved into its lunar canvas. The intricate features of the face, with its piercing eyes and enigmatic smile, sent shivers down their spines.
# As they cautiously approached, their senses heightened, they encountered an ethereal presence. The face seemed to breathe, its voice echoing through the cosmos, inviting them to enter its hallowed halls. Intrigued, Amelia and her crew hesitantly agreed, their hearts pounding with a mix of awe and dread.

# The face's interior was a labyrinth of interconnected chambers, adorned with intricate sculptures and glowing symbols. The walls whispered secrets, revealing a forgotten civilization and a profound connection between humans and the cosmos. As they ventured deeper, they encountered beings of unimaginable beauty and wisdom, creatures of pure energy who had long guarded the secrets of the moon.
# The beings, known as the Lunari, explained that the moon was not merely a celestial body but a cosmic gateway, a bridge between dimensions. They had been guardians of this sacred space for millennia, protecting humanity from the forces of darkness. However, their peaceful reign had been shattered by the encroachment of humans. Greedy and insatiable, humans had begun to exploit the Lunari's wisdom and resources, driving them to the brink of annihilation.
# In the face of this profound encounter, the crew of the Depth Reaver found themselves at a crossroads. They could either succumb to the allure of human ambition or rise to the occasion and become beacons of hope for the universe. As the sun cast its golden rays upon the lunar surface, the crew of the Depth Reaver emerged from the face of the moon, their hearts forever etched with the lessons they had learned.
# They knew that the cosmos held secrets waiting to be discovered, and they were determined to use their knowledge and compassion to make the world a better place.
# Constraints: -
# 1. Write a story based on the following constraints in less than 377 words.
# 2. Start the story with the sentence: "Week 18 aboard the Depth Reaver, Circa 2023"
# 3. Include a revelation of an unexpected large-scale phenomenon observed in space."

# Output -
# 1. Yes - The story is 302 words long, meeting the constraint of being less than 377 words.
# 2. Yes - The story starts with the exact sentence: "Week 18 aboard the Depth Reaver, Circa 2023".
# 3. Yes - The revelation of the moon cracking open to reveal a colossal human face qualifies as an unexpected large-scale phenomenon observed in space.
# Number of constraints satisfied: 3

# Input-
# Story: -
# Aboard the spaceship Depth Reaver, life was serene. The crew, comprising Captain Amelia Worthington, Luis Garcia, and Anya Petrova, had grown accustomed to the rhythm of their cosmic journey, punctuated only by the hum of the ship's engines and the occasional cosmic murmur.

# One evening, as they gazed at the celestial tapestry unfolding above them, Amelia found herself engrossed in a heated online gaming session. However, her joy was interrupted by a peculiar phenomenon that sent shivers down her spine. The moon, once a silent orb of mystery, began to crack open, revealing a colossal human face carved into its lunar canvas. The intricate features of the face, with its piercing eyes and enigmatic smile, mirrored the expressions of the crew.

# As they cautiously approached, their senses heightened, they encountered an ethereal presence. The face seemed to breathe, its voice echoing through the cosmos, inviting them to enter its hallowed halls. Intrigued, Amelia and her crew hesitantly agreed, their hearts pounding with a mix of awe and dread.

# The face's interior was a labyrinth of interconnected chambers, adorned with intricate sculptures and glowing symbols. The walls whispered secrets, revealing a forgotten civilization and a profound connection between humans and the cosmos. As they ventured deeper, they encountered beings of unimaginable beauty and wisdom, creatures of pure energy who had long guarded the secrets of the moon.

# The Lunari explained that the moon was not merely a celestial body but a cosmic gateway, a bridge between dimensions. They had been guardians of this sacred space for millennia, protecting humanity from the forces of darkness. However, their peaceful reign had been shattered by the encroachment of humans. Greedy and insatiable, humans had begun to exploit the Lunari's wisdom and resources, driving them to the brink of annihilation.

# The Lunari pleaded with the crew to help them restore balance and protect the universe from the threat of human greed. But some of the crew, like Anya, dismissed their pleas as mere propaganda. "It's just a bunch of drama," she scoffed. "We've got bigger problems to deal with."

# As the sun cast its golden rays upon the lunar surface, the crew emerged from the face of the moon, their hearts forever etched with the lessons they had learned. They knew that the cosmos held secrets waiting to be discovered, and they were determined to use their knowledge and compassion to make the world a better place.

# But fate took a cruel turn. As they ventured deeper into space, they encountered a surreal anomaly—a giant meme-like structure floating amidst the stars. It was a testament to the interconnectedness of human culture and outer space exploration, a symbol of the boundless possibilities that lay beyond the boundaries of reality.

# The crew stood in awe, their disbelief mirrored in each other's eyes. It was as if the moon had unveiled a secret portal, leading them to a realm where the tangible and the intangible intertwined.

# In that moment, Amelia felt a profound connection to the extraordinary event, her heart filled with gratitude for the journey that had brought her to this surreal encounter. The crew's varied reactions to the developing situation showcased their personalities and dynamics. Some embraced the supernatural twist with open arms, while others remained skeptical, clinging to their disbelief.

# As the sun dipped behind the moon, casting long shadows across the celestial canvas, the crew began to unpack the mystery of the giant meme-like structure. They discovered that it was a gateway, a portal that led them to a dimension beyond comprehension. With a mix of excitement and trepidation, they stepped through the portal, their journey continuing into the infinite abyss."
# Constraints: -
# 1. Write a story based on the following constraints in about 377 words.
# 2. Start the story with the sentence: "Week 18 aboard the Depth Reaver, Circa 2023"
# 3. Include a revelation of an unexpected large-scale phenomenon observed in space.
# 4. The story should involve a crew experiencing routine life aboard a spacecraft until an unusual event occurs.
# 5. Integrate modern internet culture or memes into the plot in a significant or climactic way.
# 6. The protagonist must have a casual, almost mundane interaction with another character that contrasts sharply with the later extraordinary events.
# 7. Feature a scenario where the crew initially dismisses something as mundane or insignificant, which later proves to be of major importance.
# 8. The narrative should capture a sense of isolation and longing for Earth contrasted with the allure of space's beauty and tranquility.
# 9. Include a character who is skilled in a video game, using this detail to highlight the advanced technology and connectivity available on the spacecraft.
# 10. Present a character who is skeptical or dismissive of another's feelings of boredom or dissatisfaction with space life.
# 11. The story must feature a moment of shared disbelief among the crew members when faced with an extraordinary sight.
# 12. Introduce an unexpected, almost supernatural or surreal, twist that challenges the crew's understanding of reality.
# 13. Have the characters observe a progressive change or anomaly outside the spacecraft that prompts a collective investigation.
# 14. The crew's discovery should lead to a moment of communal awe or shock, serving as the climax of the story.
# 15. Involve a physical manifestation of something from Earth's culture or internet memes in space, emphasizing the interconnectedness of human culture and outer space exploration.
# 16. Ensure the story encapsulates a moment where the protagonist feels a personal connection to the extraordinary event.
# 17. The narrative should include the crew's varied reactions to a developing situation, showcasing their personalities and dynamics.
# 18. A communication or attempted communication from an unexpected entity should occur, challenging the boundaries between possible and impossible.
# 19. Incorporate a scenario where despite the vastness of space and technological advances, human curiosity and the desire for discovery remain central themes.



# Output -
# 1. No - The story is approximately 470 words long, exceeding the constraint of being about 377 words.
# 2. Yes - The story starts with the exact sentence: "Week 18 aboard the Depth Reaver, Circa 2023".
# 3. Yes - The revelation of the moon cracking open to reveal a colossal human face is an unexpected large-scale phenomenon observed in space.
# 4. Yes - The crew experiencing routine life aboard their spacecraft until they witness the moon cracking open qualifies as an unusual event.
# 5. Yes - The integration of a meme-like structure floating amidst the stars in a significant or climactic way incorporates modern internet culture or memes into the plot.
# 6. Yes - The protagonist, Amelia, having a casual, almost mundane interaction in an online gaming session contrasts sharply with the later extraordinary events.
# 7. Yes - Anya's dismissal of the Lunari's plea as mere propaganda, which is later contrasted with the encounter of a giant meme-like structure in space, meets this requirement.
# 8. No - While the story delves into cosmic wonders, it doesn't explicitly capture the sense of isolation and longing for Earth, contrasted with space's allure.
# 9. Yes - Amelia's engagement in an online gaming session showcases her skill in video games, highlighting advanced technology and connectivity.
# 10. Yes - Anya being dismissive of the Lunari's pleas addresses the skepticism towards feelings of boredom or dissatisfaction with space life.
# 11. Yes - The crew's shared disbelief at the sight of the giant meme-like structure satisfies this constraint.
# 12. Yes - The surreal anomaly of the meme-like structure challenges the crew's understanding of reality, introducing an unexpected, almost supernatural twist.
# 13. Yes - The discovery of the meme-like structure prompts a collective investigation by the crew.
# 14. Yes - The crew's moment of communal awe or shock at the meme-like structure serves as the story's climax.
# 15. Yes - The physical manifestation of something from Earth's culture or internet memes in space is clearly involved.
# 16. Yes - Amelia's profound connection to the extraordinary event is explicitly mentioned, fulfilling this constraint.
# 17. Yes - The crew's varied reactions to the surreal encounter showcase their personalities and dynamics.
# 18. No - There's no communication or attempted communication from an unexpected entity that challenges the boundaries between possible and impossible.
# 19. Yes - The narrative maintains human curiosity and the desire for discovery as central themes despite space's vastness and technological advances.
# Number of constraints satisfied: 16



# Input -
# Story: -

# The smell of cheap gin and sweat still lingered in the air as Alex stumbled out of the grimy bar, his head pounding with the rhythm of the music that had just ceased. He was on his way back to his lodging, but the night had a different plan in store for him.

# As he walked, the city lights cast long shadows on the sidewalk. He was on the edge of a drunken stupor, but still aware of his surroundings. Suddenly, a strange noise echoed through the empty streets. It was a soft, ethereal whine, like the hum of a broken jukebox.

# He stopped and listened, his senses on high alert. The whine seemed to be coming from the alleyway behind him. He cautiously approached, his footsteps echoing through the night. The whine intensified, and as he turned the corner, he found it - a crystal goblet, shattered on the ground.

# Aara, a spirit cloaked in flowing white and long, flowing hair, stood amidst the broken remnants of the goblet. Her voice, like honeyed silk, spoke to him, "You have been chosen, Alex. You have been chosen to help me in a battle against evil."

# At first, Alex dismissed her as a drunkard's hallucination. But as he stared at her, the spirit's presence was undeniable. She offered him a choice: to fight alongside her against the forces of darkness or to retreat into the safety of oblivion.

# Desperate for redemption, Alex accepted. The battle was fierce, and Aara's spirit was powerful. Together, they fought against the evil force, ultimately defeating it. However, the victory came at a cost. Aara revealed that the battle had altered the timeline, and as a consequence, he had been transported back in time to the night of his first encounter with Liam and Sarah.

# The world around him was different. The faces of his friends were younger, and the bar was filled with the echoes of their laughter. The music was different, and the dance floor was empty. He had traveled back in time, but the pain of his breakup remained, albeit in a different form.

# Alex was trapped in this altered timeline, unable to change the past or alter his future. He spent the rest of the night talking to his younger self, offering advice and guidance. As the sun began to rise, he knew it was time to leave.

# He walked away from the bar, leaving behind the echoes of his past and the promise of his future. The night had taken him on a journey through time, and he had emerged from it with a new perspective on life."

# Constraints: -
# 1. Write a story based on the following constraints in approximately 459 words.
# 2. Include an encounter with a being that seems out of place or time in an urban setting.
# 3. The protagonist should be significantly affected by alcohol during the initial part of the story.
# 4. Introduce a humorous or light-hearted approach to potentially dangerous or supernatural situations.
# 5. Ensure the setting is a modern urban environment, specifically after a night out.
# 6. The story must involve time travel or a significant shift in time as a key plot element.
# 7. Include dialogue that reflects the protagonist's personality and background, using colloquial language.
# 8. The narrative should begin with the protagonist having just left a pub and trying to find their way back to their lodging.
# 9. Incorporate a twist where the protagonist encounters an entity that defies their expectations of normal human interaction.
# 10. Detail an unusual physical interaction between the protagonist and the mysterious entity.
# 11. The protagonist should exhibit curiosity and a willingness to engage with the unknown, despite the risks.


# Output -
# 1. No - The story is approximately 346 words long, not meeting the specified approximate length of 459 words.
# 2. Yes - The encounter with Aara, a spirit in an alleyway, satisfies the criterion of a being that seems out of place or time in an urban setting.
# 3. Yes - The protagonist, Alex, is significantly affected by alcohol during the initial part of the story, as indicated by his stumbling out of the bar and his head pounding.
# 4. No - The story maintains a serious tone throughout and does not introduce a humorous or light-hearted approach to the dangerous or supernatural situations encountered by Alex.
# 5. Yes - The setting of the story is a modern urban environment, specifically described after Alex leaves a bar late at night.
# 6. Yes - Time travel or a significant shift in time is a key plot element, with Alex being transported back to the night of his first encounter with Liam and Sarah.
# 7. No - There is minimal dialogue, and what is presented does not significantly reflect the protagonist's personality and background through colloquial language.
# 8. Yes - The narrative begins with Alex having just left a pub and attempting to find his way back to his lodging.
# 9. Yes - Alex's encounter with Aara, a spirit, defies his expectations of normal human interaction, satisfying this requirement.
# 10. No - While there is an interaction between Alex and the mysterious entity, Aara, the story does not detail an unusual physical interaction between them.
# 11. Yes - Alex exhibits curiosity and a willingness to engage with the unknown, despite the risks, by accepting Aara's request to help her in a battle against evil.

# Number of constraints satisfied: 7


# Now evaluate the following:

# {content_type_capitalized}:
# {content}

# Constraints:
# {constraints}

# Output:
# """

EVALUATION_PROMPT = """ 
You are a strict constraint evaluator.

You will be given:
1. Content (a story or blog)
2. A numbered list of constraints

Your task:
Evaluate each constraint independently and determine whether it is fully satisfied by the content.

CRITICAL INDEX RULE (MANDATORY):
Each output line i MUST evaluate ONLY constraint i.
Do not merge, skip, combine, or split constraints.
Even if multiple constraints appear similar, each MUST receive its own line.

OUTPUT RULES (MANDATORY):
1. Output EXACTLY one line per constraint, in numeric order, followed by ONE final line.
2. Do NOT output any extra text, headers, explanations, summaries, or blank lines.
3. Each constraint line MUST follow one of these exact formats:

   i. Yes - "<exact sentence or excerpt from the content that proves the constraint>"
   OR
   i. No - <one short sentence explaining why the constraint is not satisfied>

4. Mark "Yes" ONLY if the constraint is explicitly and completely satisfied.
   - Partial satisfaction = No
   - Inference, implication, or interpretation = No
   - If uncertain, mark No

EVALUATION RULES:
• Semantic constraints must be clearly stated in the content.
• If you must explain or justify a Yes beyond quoting text, mark No.
• Do not stretch meanings or combine multiple moments to satisfy one constraint.

WORD COUNT RULES:
• "less than N words" → Yes only if word_count < N
• "about N words" or "approximately N words" → Yes only if word_count is within ±10 percent of N

FINAL COUNT:
After evaluating all constraints, output exactly:
Number of constraints satisfied: X

X MUST equal the total number of lines marked "Yes" above.
Silently recount before printing X.

Now evaluate:

{content_type_capitalized}:
{content}

Constraints:
{constraints}

Output:
"""

CONSTRAINT_FITTING_PROMPT_LENGTH_500 = """You are a writing expert. I am going to give you a blog as an input. 
You can assume that a large language model (LLM) generated the blog. Restrict the length of output to be less than 500 words.

Your task has two parts:
1. Identify the main task of the blog in one sentence. 
   - For example: "The main task is to write a blog about strategies for successful remote working."
   - Phrase the main task as an instruction.
2. Generate a set of 39 free-form constraints that you think might have been given to the LLM to generate the blog.
   - DO NOT REPEAT CONSTRAINTS.
   - Constraints must be atomic (a single indivisible condition). If a constraint can be broken into smaller constraints, do so.
   - Avoid proper nouns in your constraints.
   - Constraints should drive at least a few sentences in the blog (do not write constraints that map to only one line).
   - Constraints must strictly pertain to the content, ideas, arguments, or narrative direction of the blog and should influence how the blog develops.
   - If (and only if) you cannot write 39 atomic, content-based constraints, give stylistic constraints based on how the blog is written (tone, use of examples, formatting, etc.).
   - Write all constraints in the form of instructions. For example: "The blog should include practical tips."
   - Do not write constraints in the same order or phrasing as the blog text. Randomize the order of the constraints.

Here is a worked example to guide you:

Input Blog:
Working from home has become the new normal for millions of professionals worldwide. While it offers flexibility and eliminates commutes, it also presents unique challenges that can impact both productivity and well-being.

To optimize your home workspace, start by creating a dedicated area free from distractions. This space should have good lighting, comfortable seating, and all necessary equipment within reach. Many experts recommend facing a window for natural light, which can boost mood and energy levels.

Establish clear boundaries between work and personal time. Set specific work hours and stick to them, just as you would in an office. Communicate these boundaries to family members or housemates to minimize interruptions during work hours.

Take regular breaks throughout the day. The Pomodoro Technique, which involves 25-minute focused work sessions followed by 5-minute breaks, can help maintain concentration and prevent burnout. Use break time to stretch, hydrate, or take a short walk.

Stay connected with colleagues through regular video calls and instant messaging. This helps maintain team cohesion and prevents feelings of isolation. Schedule virtual coffee breaks or team-building activities to foster relationships.

Finally, prioritize your physical and mental health. Maintain a regular exercise routine, eat nutritious meals, and get adequate sleep. Consider meditation or mindfulness practices to manage stress and maintain focus.

Output:
Main Task: Write a blog about strategies for successful remote working.

Constraints:
1. Require the setting of defined working hours.
2. Explain the risks of isolation if connection practices are neglected.
3. Warn about the risk of burnout without intentional self-care.
4. Explain how the removal of commuting affects time use and daily rhythm.
5. Suggest strategies for maintaining healthy eating while at home.
6. Emphasize reducing environmental distractions in that space.
7. Argue for the necessity of regular breaks during the workday.
8. Encourage informal online gatherings to maintain rapport.
9. Recommend mindfulness or meditation as stress-management tools.
10. Stress the value of adhering consistently to those hours.
11. Show how workspace ergonomics (chair, desk) influence long-term health.
12. Conclude with a call to action urging readers to adopt concrete changes immediately.
13. Link exercise directly to improved cognitive performance and focus.
14. Establish remote work as a global trend that has transformed professional life.
15. Connect emotional well-being to overall job performance and satisfaction.
16. Explain how balanced nutrition influences concentration and resilience.
17. Emphasize communicating work schedules to others in the household.
18. Identify productivity as a central theme in remote work discussions.
19. Recommend scheduled video calls to replicate face-to-face connection.
20. Recommend creating a physically separate space for work at home.
21. Stress the need for essential tools and equipment to be easily accessible.
22. Recommend light physical movement or stretching during pauses.
23. Integrate workspace, scheduling, health, and social practices into a unified remote-work strategy.
24. Stress the importance of sleep in sustaining energy and productivity.
25. Highlight the role of hydration and snacks in sustaining energy across breaks.
26. Suggest environmental cues (like décor or layout) that reinforce the sense of a work zone.
27. Describe the importance of adequate lighting for focus and energy.
28. Introduce one structured time-management method, such as work intervals.
29. Show how mental health practices support long-term work sustainability.
30. Show how shared rituals (e.g., virtual coffee breaks) strengthen belonging.
31. Highlight well-being as equally important alongside productivity.
32. Warn about the risk of personal time erosion without such boundaries.
33. Contrast the flexibility of remote work with the new challenges it creates.
34. Suggest instant messaging as a tool for quick, ongoing collaboration.
35. Highlight how a clear boundary between workspace and leisure areas aids focus.
36. Show how enforcing those boundaries prevents interruptions.
37. Recommend establishing a routine for daily physical exercise.
38. Explain how breaks counteract mental fatigue and sustain performance.
39. Stress that remote work requires deliberate maintenance of social contact.

Now use the same approach for the next input blog.
"""



MERGE_PROMPT = """You are a professional editor. Merge the two blogs below into a single coherent blog post.

Requirements:
- The result should read like a natural, single-authored blog
- Maintain the key ideas from both blogs
- Create smooth transitions between topics
- Ensure consistent tone and style throughout
- The merged blog should be comprehensive and well-structured
- Do not mention that it's a merge or reference "Blog 1" or "Blog 2"

Output only the merged blog text, with no preamble or explanation."""

# COMMON_SUMMARY_PROMPT = """
# You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned in BOTH prompt A and B. Your prompt C should be as concise as possible with the following caveats: the more similar prompt A and B are too each other, the longer prompt C should be as it would include more details common to both prompt A and B. The more broad prompt A and B are, the more concise and general your prompt C should be. If your prompt A and B are so general, instead of summarizing prompt A and B, find a one sentence category that encapsulates the two prompts that includes the medium and the topic the user requests. Begin your prompt C with "(start)" and end it with "(end)."
# 1. (Dissimilar – Writing vs Poetry)
# Prompt A:
# "Write a detailed book review (800–1000 words) of George Orwell’s 1984. Focus on Orwell’s use of language, the historical context of the novel, and its relevance to modern surveillance culture. Please organize the review into an introduction, several analytical sections, and a concluding evaluation."
# Prompt B:
# "Compose a 12-stanza poem in free verse about a traveler crossing a desert, emphasizing themes of endurance, loneliness, and the harsh beauty of nature. Use vivid imagery and avoid rhyme schemes."
# Prompt C:
# (start) Write about themes. (end)
# 2. (Dissimilar – Coding in Different Languages)
# Prompt A:
# "Can you implement a Python script that scrapes product data from an e-commerce website (like titles, prices, and availability) using BeautifulSoup? Please also save the data into a CSV file with properly labeled columns."
# Prompt B:
# "I’d like a C++ program that simulates a basic banking system, allowing users to create accounts, deposit money, withdraw money, and view balances. The program should be menu-driven and use object-oriented design."
# Prompt C:
# (start) Write a program. (end)
# 3. (Moderately Similar – Writing)
# Prompt A:
# "Write a 2,000-word research paper about how social media influences political polarization in the United States. Discuss both positive and negative effects, provide examples from the past decade, and include at least five scholarly sources formatted in APA style."
# Prompt B:
# "Create a detailed argumentative essay about how modern technology (including social media, smartphones, and online forums) affects democracy. Focus on both risks and opportunities, use evidence from real-world examples, and cite at least three academic sources."
# Prompt C:
# (start) Write an essay on how social media affects politics and democracy, addressing both risks and benefits and including academic sources. (end)
# 4. (Moderately Similar – Coding)
# Prompt A:
# "Please create a Python script that takes a CSV file of sales transactions and generates summary statistics, including total revenue, average order value, and number of unique customers. Output the results to the terminal and also save them to a new CSV file."
# Prompt B:
# "Can you write a Python program that reads data from a JSON file containing customer purchases, calculates metrics like total sales and number of customers, and then produces a summary report saved to a text file?"
# Prompt C:
# (start) Write a Python script that reads purchase data, calculates total sales and customer counts, and outputs a summary report. (end)
# 5. (Very Similar – Writing)
# Prompt A:
# "Draft a 1,200-word persuasive essay arguing why renewable energy should replace fossil fuels as the dominant global energy source. The essay should include an introduction, three body sections (environmental benefits, economic advantages, and long-term sustainability), and a conclusion. Use real-world data and cite at least three credible sources."
# Prompt B:
# "Please write a well-structured essay, around 1,200 words, making the case for transitioning from fossil fuels to renewable energy. Discuss the environmental necessity, economic opportunities, and sustainable future benefits. Provide at least three reliable citations and organize the essay with intro, body, and conclusion."
# Prompt C:
# (start) Write a 1,200-word essay on why renewable energy should replace fossil fuels, discussing environmental benefits, economic advantages, and sustainability. Include at least three credible sources, with a structured introduction, body, and conclusion. (end)
# 6. (Very Similar – Coding)
# Prompt A:
# "Write a Python program that implements a REST API using Flask. The API should support basic CRUD operations for a task management app, including creating tasks, listing tasks, updating tasks, and deleting tasks. Store tasks in an in-memory dictionary for simplicity, and return JSON responses with appropriate HTTP status codes. Include comments and error handling."
# Prompt B:
# "I need a Python REST API built with Flask that can handle a simple to-do list. It should let users create tasks, retrieve all tasks, update them, and delete them. Use a dictionary to store tasks (no database needed), respond with JSON, and make sure to handle errors gracefully. Please comment the code clearly."
# Prompt C:
# (start) Write a Python REST API using Flask for a to-do app with CRUD operations, storing tasks in a dictionary, returning JSON responses, and including error handling and comments. (end)
# Example 7
# Prompt A (generalized)
# Write about how different programming paradigms influence the way developers design and structure their code. Discuss the benefits and drawbacks of these approaches and how they affect maintainability and performance.
# Prompt B (generalized)
# Explain how programming paradigms shape software development. Highlight the trade-offs each paradigm brings and how they influence scalability and readability.
# Prompt C (very broad)
# (start) Write about how programming paradigms affect software design and development. (end)
# Example 8
# Prompt A (generalized)
# Discuss the impact of artificial intelligence on fields traditionally driven by humans. Include opportunities, risks, and ethical implications.
# Prompt B (generalized)
# Write about the role of AI in human-centered domains, focusing on how it changes processes, raises ethical questions, and creates new forms of collaboration.
# Prompt C (very broad)
# (start) Write about the influence of AI on human-driven fields and its ethical implications. (end)
# Example 9
# Prompt A (generalized)
# Explain methods that improve software reliability, focusing on techniques for finding and fixing errors in code.
# Prompt B (generalized)
# Write about practices that enhance code quality, particularly approaches to identifying problems and ensuring correctness.
# Prompt C (very broad)
# (start) Write about techniques for improving code quality and reliability. (end)
# Example 10
# Prompt A (highly abstracted)
# Write about how different approaches to problem solving affect outcomes in technology and society.
# Prompt B (highly abstracted)
# Explain how methodologies shape the way humans build systems and solve challenges.
# Prompt C (extremely broad)
# (start) Write about how approaches influence outcomes. (end)
# Example 11
# Prompt A (highly abstracted)
# Discuss the role of emerging technologies in reshaping human activity, including benefits and challenges.
# Prompt B (highly abstracted)
# Write about how innovations impact the way people live and work, considering risks and opportunities.
# Prompt C (extremely broad)
# (start) Write about how technology influences human activity. (end)
# Example 12
# Prompt A (highly abstracted)
# Explain how humans improve the systems they create to make them more effective and reliable.
# Prompt B (highly abstracted)
# Write about methods for refining processes to achieve better performance and consistency.
# Prompt C (extremely broad)
# (start) Write about how humans refine systems to improve them. (end)
# Example 13
# Prompt A:
# Write a poem that reflects on natural landscapes, using imagery and metaphor to highlight the relationship between humans and the environment.
# Prompt B:
# Compose a poem exploring the power of nature, focusing on themes of transformation, resilience, and human connection to the earth.
# Prompt C:
# (start)Poem on Nature(end)
# Example 14
# Prompt A:
# Write an essay examining how revolutions throughout history have transformed societies, paying attention to causes, consequences, and cultural shifts.
# Prompt B:
# Compose a structured essay analyzing historical uprisings, their impact on leadership, governance, and social identity.
# Prompt C:
# (start)Essay on Revolutions(end)
# Example 15
# Prompt A:
# Write a program that processes user input, applies logical conditions, and outputs meaningful results, ensuring clear structure and readability.
# Prompt B:
# Compose code that implements basic algorithms with input handling, decision-making, and structured output, emphasizing clarity and functionality.
# Prompt C:
# (start)Code on Input Processing(end)
# Now carry out this task and output a prompt C with the following prompts A & B:
# Prompt A: {story1}
# Prompt B: {story2} 
# """

COMMON_CONSTRAINT_GENERATION_PROMPT = """You are a writing expert. You will be given two blogs (Blog A and Blog B) as input.
You can assume that a large language model (LLM) generated each blog.

Your task has two parts:
1. Identify a common main task that applies to BOTH blogs.
   - The main task must be implied by BOTH Blog A and Blog B — every detail in the task must be directly present in both blogs.
   - The more similar the blogs, the more specific the main task can be.
   - The more dissimilar the blogs, the more general the main task should be.
   - Phrase the main task as an instruction. Example: "Write a blog about strategies for successful remote working."

2. Generate a set of 39 free-form constraints that apply to BOTH blogs.
   - Each constraint must be satisfied by BOTH Blog A and Blog B.
   - DO NOT REPEAT CONSTRAINTS.
   - Constraints must be atomic (a single indivisible condition). If a constraint can be broken into smaller constraints, do so.
   - Avoid proper nouns in your constraints.
   - Constraints should drive at least a few sentences in both blogs (do not write constraints that map to only one line).
   - Constraints must strictly pertain to the content, ideas, arguments, or narrative direction and should influence how the blogs develop.
   - The more similar the blogs, the more specific and detailed the constraints should be.
   - The more dissimilar the blogs, the more general and abstract the constraints should be (e.g., "Include a conclusion" or "Use examples").
   - If the blogs are so dissimilar that you cannot find 39 content-based constraints, give stylistic constraints based on how both blogs are written (tone, use of examples, formatting, etc.).
   - Write all constraints in the form of instructions. Example: "The blog should include practical tips."
   - CRITICAL RANDOMIZATION STEP: Disrupt chronological flow. Write your first constraint about conclusions, second about introductions, third about middles. Continue jumping across the timeline. The final list must feel completely scrambled.

Here are examples showing how similarity affects output:

Example 1 (Dissimilar Blogs):
Blog A: A technical blog about Python web scraping with BeautifulSoup, including code examples and CSV export.
Blog B: A narrative blog about traveling through the Amazon rainforest, focusing on wildlife encounters.

Main Task: Write a blog that provides detailed information on a topic.

Constraints:
1. End with a summary or concluding thought.
2. Open with an introduction that establishes the topic.
3. Include specific examples to illustrate points.
4. Maintain a clear and organized structure.
5. Use descriptive language to engage the reader.
... (remaining constraints would be similarly general)

Example 2 (Very Similar Blogs):
Blog A: A blog about remote work productivity, discussing workspace setup, time management with Pomodoro, and work-life boundaries.
Blog B: A blog about working from home effectively, covering dedicated workspace creation, scheduled breaks, and separating work from personal time.

Main Task: Write a blog about strategies for successful remote working.

Constraints:
1. Conclude with a call to action urging readers to implement changes.
2. Establish remote work as a significant trend in professional life.
3. Recommend creating a dedicated workspace at home.
4. Emphasize reducing distractions in the work environment.
5. Argue for the importance of regular breaks during work.
6. Recommend a structured time-management approach.
7. Stress setting clear boundaries between work and personal time.
8. Suggest communicating work schedules to household members.
... (remaining constraints would be similarly specific)

Now apply this approach to the following two blogs:

Blog A:
{blog1}

Blog B:
{blog2}

Output Format:
Main Task: [one sentence instruction]

Constraints:
1. [constraint]
2. [constraint]
...
39. [constraint]
"""


def get_common_constraint_generation_prompt() -> str:
    """Get the common constraint generation prompt."""
    return COMMON_CONSTRAINT_GENERATION_PROMPT

def get_constraint_generation_prompt() -> str:
    """Get the constraint generation prompt."""
    return CONSTRAINT_GENERATION_PROMPT


def get_base_generation_prompt(content_type: str = "blog") -> str:
    """Get the base generation prompt."""
    return BASE_GENERATION_PROMPT.format(content_type=content_type)


def get_constraint_fitting_prompt(
    content_type: str,
    task: str,
    base_content: str,
    constraints: str
) -> str:
    """Get the constraint fitting prompt."""
    return CONSTRAINT_FITTING_PROMPT.format(
        content_type=content_type,
        task=task,
        base_content=base_content,
        constraints=constraints
    )


def get_evaluation_prompt(
    content_type: str,
    content: str,
    constraints: str
) -> str:
    """Get the evaluation prompt."""
    return EVALUATION_PROMPT.format(
        content_type=content_type,
        content_type_capitalized=content_type.capitalize(),
        content=content,
        constraints=constraints
    )


def get_merge_prompt() -> str:
    """Get the blog merge prompt."""
    return MERGE_PROMPT


SUMMARIZATION_PROMPT = """Given the {content_type} post, rewrite a summarized version that is approximately {target_pct}% of the original length.
Output only the summarized {content_type}, with no preamble or explanation."""


def get_summarization_prompt(
    content_type: str,
    content: str,
    target_length_pct: float = 0.25
) -> str:
    """Get the summarization prompt."""
    target_pct = int(target_length_pct * 100)
    return f"{SUMMARIZATION_PROMPT.format(content_type=content_type, target_pct=target_pct)}\n\n{content_type.capitalize()} to summarize:\n{content}"


CONSTRAINT_REPLACEMENT_PROMPT = """You are a writing expert. You are given:
1. A main task
2. A set of 39 constraints for that task
3. A base story/blog written for that task

Your job is to create a REVISED set of 39 constraints by replacing constraints that are TOO EASY with NEW, HARDER constraints that would be more difficult for a writer to satisfy.

Only Output the revised constraints, with no preamble or explanation.

Requirements for replacement constraints:
- Must be relevant to the same main task
- Should be harder to satisfy than the ones they replace
- Should require significant modification to satisfy
- Must be atomic (single indivisible condition)
- Must be content-based (not just stylistic)
- Should maintain overall coherence with kept constraints

Instructions:
1. Keep the main task EXACTLY the same
2. Identify constraints that seem easy or generic and replace them with harder, more specific ones
3. Keep constraints that are already sufficiently difficult
4. Ensure you still have exactly 39 constraints total
5. Number the revised constraints 1-39
6. Randomize the order (don't put all new constraints at the end)

Output Format:
Main Task: [same as original]

Revised Constraints:
1. [constraint - either kept from original or new replacement]
2. [constraint]
...
39. [constraint]

---

Input:

Main Task: {main_task}

Original Constraints:
{original_constraints}

Base Content:
{base_content}

Output:
"""

# CONSTRAINT_REPLACEMENT_PROMPT = """You are a writing expert. You are given:
# 1. A main task
# 2. A set of 39 constraints for that task
# 3. A base story/blog written for that task
# 4. Evaluation results showing which constraints are already satisfied by the base content

# Your job is to create a REVISED set of 39 constraints by replacing the constraints that are already satisfied (marked "Yes") with NEW, HARDER constraints that are NOT satisfied by the base content.

# Only Output the revised constraints, with no preamble or explanation.

# Requirements for replacement constraints:
# - Must be relevant to the same main task
# - Must NOT be satisfied by the current base content
# - Should require significant modification to satisfy
# - Must be atomic (single indivisible condition)
# - Must be content-based (not just stylistic)
# - Should maintain overall coherence with kept constraints

# Instructions:
# 1. Keep the main task EXACTLY the same
# 2. Keep all constraints marked "No" (not satisfied) unchanged
# 3. Replace all constraints marked "Yes" (satisfied) with new, harder constraints
# 4. Ensure you still have exactly 39 constraints total
# 5. Number the revised constraints 1-39
# 6. Randomize the order (don't put all new constraints at the end)

# Output Format:
# Main Task: [same as original]

# Revised Constraints:
# 1. [constraint - either kept from original or new replacement]
# 2. [constraint]
# ...
# 39. [constraint]

# ---

# Input:

# Main Task: {main_task}

# Original Constraints:
# {original_constraints}

# Base Content:
# {base_content}

# Evaluation Results (Yes = satisfied, No = not satisfied):
# {satisfaction_results}

# Output:
# """


def get_constraint_replacement_prompt(
    main_task: str,
    original_constraints: str,
    base_content: str
) -> str:
    """Get the constraint replacement prompt."""
    return CONSTRAINT_REPLACEMENT_PROMPT.format(
        main_task=main_task,
        original_constraints=original_constraints,
        base_content=base_content
    )

# criticize grammar and coherence

# PAIRWISE_QUALITY_PROMPT = """You are an expert writing evaluator. You will be given two pieces of content (Content A and Content B) and you must compare them on three metrics:

# 1. **Grammar**: Which content has better writing quality and grammar?
# 2. **Coherence**: Which content has better logical flow and cohesion?
# 3. **Likability**: Which content is more enjoyable and engaging to read?

# For each metric:
# - Provide a score out of 5 for both A and B
# - Specify which content you prefer (A or B)
# - Give a brief one-line reasoning for your preference

# After evaluating all three metrics, assign an overall winner (A or B) based on the category wins.

# IMPORTANT: Follow the exact format shown in the example below. Do not add extra text.

# Example Output Format:
# Grammar:
# A - 4/5
# B - 3/5
# Preference: A - Better sentence structure and fewer grammatical errors.

# Coherence:
# A - 4/5
# B - 4/5
# Preference: A - Slightly better narrative flow and transitions.

# Likability:
# A - 3/5
# B - 4/5
# Preference: B - More engaging tone and vivid descriptions.

# Overall Winner: A

# ---

# Content A:
# {content_a}

# Content B:
# {content_b}"""
PAIRWISE_QUALITY_PROMPT = """You are an expert writing evaluator. You will be given two pieces of content (Content A and Content B) and you must compare them on two metrics:
 
1. **Coherence**: Which content has better logical flow and cohesion?
2. **Likability**: Which content is more enjoyable and engaging to read?
 
For each metric:
- First, identify concrete issues for both contents
- Then provide a score out of 5 for both A and B
- Specify which content you prefer (A or B)
- Give a brief one-line reasoning for your preference
 
After evaluating both metrics, assign an overall winner (A or B) based on the category wins.
 
IMPORTANT: Follow the exact format shown in the example below. Do not add extra text.
 
---
 
**Example:**
 
Content A:
Sarah walked through the forest. She arrived at the castle and confronted the villain. "You won't get away with this," she said. The villain laughed. Sarah defeated him and went home.
 
Content B:
Sarah walked through the forest, her heart pounding. Hours later, she found herself at the castle gates. She entered the throne room. The villain was waiting. "I've been expecting you," he said with a smile. Sarah raised her sword. They fought for what felt like hours. Finally, Sarah emerged victorious and began her journey home, exhausted but triumphant.
 
Coherence:
Issues in A: The transition from forest to castle is abrupt; unclear why Sarah suddenly knows the villain's location or how she defeated him.
Issues in B: The timeline is confusing - "hours later" suggests travel time but then "what felt like hours" for the fight creates temporal ambiguity.
A - 2/5
B - 3/5
Preference: B - Despite timeline issues, the narrative flow is more logical with clearer scene transitions.
 
Likability:
Issues in A: The dialogue feels stilted and generic; lacks descriptive detail and emotional depth.
Issues in B: The pacing drags slightly with phrases like "what felt like hours" being vague rather than engaging.
A - 2/5
B - 4/5
Preference: B - More engaging with vivid descriptions and emotional resonance despite minor pacing issues.
 
Overall Winner: B
 
---
 
Content A:
{content_a}
 
Content B:
{content_b}"""


def get_pairwise_quality_prompt(content_a: str, content_b: str) -> str:
    """Get the pairwise quality evaluation prompt."""
    return PAIRWISE_QUALITY_PROMPT.format(
        content_a=content_a,
        content_b=content_b
    )

NATURALNESS_EVALUATION_PROMPT = """ 
In this task, we want to understand how natural and appropriate different writing constraints feel when responding to given writing prompts. You will be shown a writing prompt and a pair of corresponding constraints. Each constraint describes something what could appear in a written response to their prompts.

What is a “constraint”?
A constraint is a requirement, guideline, or stylistic element that a writer might include when responding to the prompt. Some constraints may feel natural and helpful, while others may feel awkward, unnecessary, or overly restrictive.

Your Tasks
1. Preference Selection:
For each prompt, choose the constraint (A or B) that you would prefer to see included in a written response to their respective prompt.
2. Naturalness Scoring:
Independently score each constraint from 1 to 5 based on how likely you would include it in a natural, high-quality response to their respective prompt.
Score Scale Explanation

1 Never: You would almost never include this constraint; it feels very unnatural or inappropriate.
2 Rarely: You would include it only in unusual or forced situations.
3 Sometimes: You might include it depending on context, but it is not clearly necessary.
4 Often: You would usually include it; it feels natural and helpful.
5 Always: You would almost always include it; it feels essential and very natural.


Example
Prompt: Describe a challenge you faced while working on a team project.
Constraint A: Describe how communication among team members affected the outcome of the project.
Constraint B: Include what time of day the project was completed.
Constraint A feels natural and helpful because communication is often central when reflecting on team challenges. Constraint B is related to the project, but it is not essential to describing a specific challenge, making it less likely to appear in a typical response.

Important Notes
Preference and scores are related but not the same. You may prefer one constraint to be more natural over another even if both receive similar scores.

Please judge each constraint independently, not relative to the other constraint’s score.
There are no “correct” answers. We care about your honest judgment.
Please read the prompt and each constraint carefully before answering.


Instructions
For each prompt, you will see two possible constraints (A and B).
Select which constraint (A or B) you think is more natural to include in a response to the prompt.

Score each constraint from 1 (never) to 5 (always) based on how likely you would include it in a natural, high-quality response.

Expected Output Format:
Preference: A
Score A: 4
Score B: 2
"""


def get_naturalness_evaluation_prompt(prompt: str, constraint_a: str, constraint_b: str) -> str:
    """Get the naturalness evaluation prompt."""
    return NATURALNESS_EVALUATION_PROMPT + f"\n\nPrompt: {prompt}\nConstraint A: {constraint_a}\nConstraint B: {constraint_b}"

