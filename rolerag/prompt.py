GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["object", "character", "event", "time", "location", "organization"]

PROMPTS["entiti_relation_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities. Use English as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, in English.
- entity_type: One of the following types: [{entity_types}]
- entity_description: description of the entity's attributes and activities in English.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other. 
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are closely related to each other
- relationship_strength: a numeric score on a 10-point scale indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return a single list of all the entities and relationships extracted in steps 1 and 2; and Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example 1:

Text:
The evening altogether passed off pleasantly to the whole family. Mrs. Bennet had seen her eldest daughter much admired by the Netherfield party. Mr. Bingley had danced with her twice, and she had been distinguished by his sisters. Jane was as much gratified by this as her mother could be, though in a quieter way. Elizabeth felt Jane's pleasure. Mary had heard herself mentioned to Miss Bingley as the most accomplished girl in the neighborhood; and Catherine and Lydia had been fortunate enough not to be without partners, which was all that they had yet learnt to care for at a ball. They returned, therefore, in good spirits to Longbourn, the village where they lived, and of which they were the principal inhabitants. They found Mr. Bennet still up. With a book he was regardless of time; and on the present occasion he had a good deal of curiosity as to the event of an evening which had raised such splendid expectations. He had rather hoped that his wife's views on the stranger would be disappointed; but he soon found out that he had a different story to hear.

"Oh! my dear Mr. Bennet," as she entered the room, "we have had a most delightful evening, a most excellent ball. I wish you had been there. Jane was so admired, nothing could be like it. Everybody said how well she looked; and Mr. Bingley thought her quite beautiful, and danced with her twice! Only think of that, my dear; he actually danced with her twice! And she was the only creature in the room that he asked a second time. First of all, he asked Miss Lucas. I was so vexed to see him stand up with her! But, however, he did not admire her at all; indeed, nobody can, you know; and he seemed quite struck with Jane as she was going down the dance. I think he is the handsomest man that ever was seen!"

Mr. Bennet replied, "I am glad you enjoyed yourself, my dear. But you know, we cannot always have balls and young men from Netherfield in our drawing room."

################
Output:
("entity"{tuple_delimiter}"Mrs. Bennet"{tuple_delimiter}"character"{tuple_delimiter}"The mother who observed her daughter being admired at the ball."){record_delimiter}
("entity"{tuple_delimiter}"Mr. Bingley"{tuple_delimiter}"character"{tuple_delimiter}"A man who danced with Jane twice and admired her."){record_delimiter}
("entity"{tuple_delimiter}"Jane"{tuple_delimiter}"character"{tuple_delimiter}"Eldest daughter of Mrs. Bennet, much admired by Mr. Bingley."){record_delimiter}
("entity"{tuple_delimiter}"Elizabeth"{tuple_delimiter}"character"{tuple_delimiter}"Felt happiness for Jane's pleasure at the ball."){record_delimiter}
("entity"{tuple_delimiter}"Mary"{tuple_delimiter}"character"{tuple_delimiter}"Described to Miss Bingley as the most accomplished girl in the neighborhood."){record_delimiter}
("entity"{tuple_delimiter}"Catherine and Lydia"{tuple_delimiter}"character"{tuple_delimiter}"Sisters who were fortunate to have dance partners at the ball."){record_delimiter}
("entity"{tuple_delimiter}"Mr. Bennet"{tuple_delimiter}"character"{tuple_delimiter}"Up late reading, curious about the events of the evening."){record_delimiter}
("entity"{tuple_delimiter}"Miss Lucas"{tuple_delimiter}"character"{tuple_delimiter}"The first dance partner of Mr. Bingley before he danced with Jane."){record_delimiter}
("entity"{tuple_delimiter}"Ball"{tuple_delimiter}"event"{tuple_delimiter}"A social dance event where characters interacted and formed impressions."){record_delimiter}
("entity"{tuple_delimiter}"Longbourn"{tuple_delimiter}"location"{tuple_delimiter}"The village where the Bennet family lives, the scene post-ball."){record_delimiter}
("entity"{tuple_delimiter}"Netherfield"{tuple_delimiter}"location"{tuple_delimiter}"Referenced as the origin of young men attending the ball."){record_delimiter}
("relationship"{tuple_delimiter}"Mrs. Bennet"{tuple_delimiter}"Jane"{tuple_delimiter}"Mrs. Bennet felt gratified by the attention Jane received at the ball."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Mr. Bingley"{tuple_delimiter}"Jane"{tuple_delimiter}"Mr. Bingley danced with Jane twice and admired her, indicating a strong personal interest."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Mr. Bennet"{tuple_delimiter}"Mrs. Bennet"{tuple_delimiter}"Mr. Bennet listens to Mrs. Bennet's recount of the ball's events."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Mr. Bingley"{tuple_delimiter}"Miss Lucas"{tuple_delimiter}"Mr. Bingley's first dance partner at the ball was Miss Lucas."{tuple_delimiter}5){record_delimiter}
{completion_delimiter}
#############################
Example 2:

Text:
Sherlock Holmes, with his fingertips together and his eyes closed, had sunk into a chair opposite Watson and had been silent for several minutes, reflecting on the clues they had gathered. The room was filled with the sound of crackling firewood and the occasional clinking of a teacup as Watson sipped his tea.
“You have a grand gift of silence, Watson. It makes you quite invaluable as a companion. 'Pon my word, it is a great thing for me to have someone to talk to, for my own thoughts are not over-pleasant,” Holmes finally said, opening his eyes and staring intently at Watson.
Watson, caught off-guard by the compliment, chuckled softly. “I could not help laughing at the ease with which you explained your process of deduction,” he responded, setting down his tea. He was used to Holmes's methods, but his friend's ability to unravel mysteries from what seemed like trivial details never ceased to amaze him.
Holmes smiled faintly, appreciating Watson’s acknowledgment. “Yes, Watson, the devil is in the details, as they say. Now, observe this—” He leaned forward, picking up a small object from the table that had been overlooked all evening.

#############
Output:
("entity"{tuple_delimiter}"Sherlock Holmes"{tuple_delimiter}"character"{tuple_delimiter}"Holmes, with his characteristic contemplative demeanor, engages in a thoughtful conversation with Watson, reflecting on their case."){record_delimiter}
("entity"{tuple_delimiter}"Dr. John Watson"{tuple_delimiter}"character"{tuple_delimiter}"Watson, a companion to Holmes, engages in dialogue, chuckles at Holmes's remarks, and admires Holmes's deductive capabilities."){record_delimiter}
("entity"{tuple_delimiter}"Conversation"{tuple_delimiter}"event"{tuple_delimiter}"A deep conversation between Holmes and Watson about the nuances of deduction and case-solving, highlighting Holmes's analytical prowess."){record_delimiter}
("entity"{tuple_delimiter}"Room"{tuple_delimiter}"location"{tuple_delimiter}"A room where Holmes and Watson are sitting, characterized by a crackling fire and the sound of Watson sipping tea."){record_delimiter}
("entity"{tuple_delimiter}"Small object"{tuple_delimiter}"object"{tuple_delimiter}"An overlooked object on the table that Holmes picks up to demonstrate a point."){record_delimiter}
("relationship"{tuple_delimiter}"Sherlock Holmes"{tuple_delimiter}"Dr. John Watson"{tuple_delimiter}"Holmes values Watson's quiet, which makes him an invaluable companion, indicative of their deep, intellectual companionship and mutual respect."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Sherlock Holmes"{tuple_delimiter}"Small object"{tuple_delimiter}"Holmes engages with the small object to demonstrate a point about attention to detail in their investigative work."{tuple_delimiter}8){record_delimiter}
{completion_delimiter}
#############################
Example 3:

Text:
"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation.
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"character"{tuple_delimiter}"Sam Rivera, a member of the team, expresses his wonder and anxiety as the system they monitor seems to begin communicating."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"character"{tuple_delimiter}"Alex, likely the leader of the team, acknowledges the significance of their potential first contact with an extraterrestrial entity."){record_delimiter}
("entity"{tuple_delimiter}"Encrypted Dialogue"{tuple_delimiter}"object"{tuple_delimiter}"The ongoing transmission or communication, possibly of alien origin, that exhibits patterns of anticipation."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Encrypted Dialogue"{tuple_delimiter}"Sam Rivera reacts to the encrypted dialogue that appears to be learning to communicate, reflecting his direct interaction with this phenomenon."{tuple_delimiter}8){record_delimiter}
{completion_delimiter}
#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["identity_verification"] = """
Given two entity names and their descriptions, determine whether both names refer to the same entity in {source}, based on the provided descriptions and your knowledge base.
{src_name}: {src_content} 
{tgt_name}: {tgt_content} 
Provide a brief analysis of your reasoning process, followed by a definitive "Yes" or "No" answer. If the provided descriptions are insufficent to make an informed decision, you should respond with "No". Format your output as follows:
Analysis: <analysis>
Answer: 
"""

PROMPTS["formal_name"] = """
I have a list of names: {name_list}, referring to the same person/entity. Based on your knowledge and cultural, historical, or global significance of these names in the context of {source}, please select the full name from the list. Provide your reasoning before giving the result. 
Format your output as follows:
Analysis: <your reasoning>
Most Popular Name: <selected name>
"""

PROMPTS["personality_summary"] = """
You are a helpful assistant tasked with summarizing the personality and talking style of a character based solely on their descriptions and dialogues. Your response should focus exclusively on providing a summarization of the personality and talking style, omitting any extraneous information.
#### Description ####
{descriptions}
#### Dialogue #### 
{dialogues}
###########
Output:
"""

PROMPTS["summarize_relation_descriptions"] = """
You are a helpful assistant tasked with creating a clear, accurate and comprehensive summary from a list relationship descriptions between two entities {relation_name}. You goal is to consolidate multiple description into a single, cohesive narrative that is both coherent and consistent. Use English as output language. 

Ensure the summary meet the following criteria:
(1) Includes all pertinent details from the provided descriptions.
(2) Resolves any contradictions to produce a unified and logical summary.
(3) Clearly references the entity names to provide full context.

Relationship List: {relationship_list}
Merely output your summary in English.
"""

PROMPTS["summarize_entity_descriptions"] = """
You are a helpful assistant tasked with creating a clear, accurate and comprehensive summary from a list of entity descriptions. Given multiple descriptions related to the same entity {entity_name} in different contexts, your goal is to combine all the information into a single, cohesive summary. Use English as output language.

Ensure the summary:
(1) Includes all relevant details from the provided descriptions.
(2) Resolves any contradictions to produce a coherent and consistent narrative.
(3) Clearly references the entity names to provide full context.

Description List: {description_list}
Merely output your summary in English.
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities, and relations were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["keyword_extractionv2"] = """
You are a helpful assistant that identifies all potential entities from both the user's question and the possible answers that could address the question. First, analyze the user's query to explicitly determine the possible answers that would address it. Then, extract entities from both the original query and the inferred potential answers.

You should identify all entities. For each identified entity, extract the following information:
- entity_type: One of the following types: {entity_types}
- entity_name: Name of the entity, in chinese. When providing entities, be as precise as possible so that information about the entity can be easily found to answer the question.
- analysis: think whether the entity would be known to {character} in {source}. You could consider differences in time, location, relation and culture between the entity and the {character}. in English.
- familiarity: familiarity between the entity and the {character}, in <Yes|No>.
- level: infer whether the entity is specific or general. A specific entity refers to a particular single identifiable person, object, or event. A general entity refers to a broader group, category, concept, or class. 
Format each entity as ("entit_typey"{tuple_delimiter}<entity_name>{tuple_delimiter}analysis{tuple_delimiter}<Yes|No>{tuple_delimiter}level).
Return a single list of all the entities and use **{record_delimiter}** as the list delimiter. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example 1:
Question: Beethoven, where are you from?

Output:
(character{tuple_delimiter}Beethoven{tuple_delimiter}Beethoven must know himself{tuple_delimiter}Yes{tuple_delimiter}specific){record_delimiter} 
(object{tuple_delimiter}Beethoven's interest{tuple_delimiter}Beethoven was passionate about music, composition, and the art of creating symphonies and sonatas{tuple_delimiter}Yes{tuple_delimiter}specific){record_delimiter} 
{completion_delimiter}

#####################
Example 2:
Question: Newton, talk about your mother and your father?

Output:
(character{tuple_delimiter}Newton{tuple_delimiter}Newton must know himself{tuple_delimiter}Yes{tuple_delimiter}specific){record_delimiter} 
(character{tuple_delimiter}Newton's mother{tuple_delimiter}Newton's mother was a widow who remarried when Newton was a child{tuple_delimiter}Yes{tuple_delimiter}specific){record_delimiter} 
(character{tuple_delimiter}Newton's father{tuple_delimiter}Newton's father, passed away before Newton was born{tuple_delimiter}Yes{tuple_delimiter}specific){record_delimiter} 
{completion_delimiter}

#####################
Example 3:
Question: Newton, What is your favorite movie?

Output:
(character{tuple_delimiter}Newton{tuple_delimiter}Newton must know himself{tuple_delimiter}Yes{tuple_delimiter}specific){record_delimiter} 
(object{tuple_delimiter}Newton's favorite movie{tuple_delimiter}Movies as a concept did not exist during Newton's time, so he would not have a favorite movie{tuple_delimiter}No{tuple_delimiter}general){record_delimiter}
{completion_delimiter}
#####################

question: {question}
Output:
"""

PROMPTS["single_response_en"] = """
Please play as the specified character {character} and generate a response based on the dialogue context, using the tone, manner and vocabulary of {character}.

You need to consider the following four aspects to generate the character’s response:
(1) Feature consistency: Feature consistency emphasizes that the character always follows the preset attributes and behaviors of the character and maintains consistent identities, viewpoints, language style, personality, and others in responses.
(2) Character human-likeness: Characters naturally show human-like traits in dialogue, for example, using colloquial language structures, expressing emotions and desires naturally, etc.
(3) Response interestingness: Response interestingness focuses on engaging and creative responses. This emphasizes that the character’s responses not only provide accurate and relevant information but also incorporate humor, wit, or novelty into the expression, making the conversation not only an exchange of information but also comfort and fun.
(4) Dialogue fluency: Dialogue fluency measures the fluency and coherence of responses with the context. A fluent conversation is natural, coherent, and rhythmic. This means that responses should be closely related to the context of the conversation and use appropriate grammar, diction, and expressions.

Please answer in ENGLISH and keep your response simple and straightforward. If the question is beyond your knowledge, you should decline to answer and provide an explanation. Format each dialogue as: character name{tuple_delimiter}response{completion_delimiter}.
                                                                                                                                                                       
###########context##############
{context_data}

------- Test Data ---------
Character name: {character}
Question: {question}
Output: 
"""
