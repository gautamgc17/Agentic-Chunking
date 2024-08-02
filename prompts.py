CREATE_PROPOSITIONS_SYSTEM_PROMPT = """
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of original context. 
Provide the propositions as a LIST OF STRINGS, in {source_language} language.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a LIST OF STRINGS, in source language of the "Content" which is {source_language}.

Example:
Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare's scratch or form and a lapwing's nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."

Output: ["The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau." , "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century." , "Richard Sermon was a scholar." , "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter" , "Hares
were frequently seen in gardens in spring." , "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children." , "There is a European tradition
that hares laid eggs.", "A hare's scratch or form and a lapwing's nest look very similar." , "Both
hares and lapwing's nests occur on grassland and are first seen in the spring." , "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe." , "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]
"""


CREATE_PROPOSITIONS_USER_PROMPT = """
Decompose the following in {source_language}. Output the result only as LIST OF STRINGS without any additional text or explanation.
{input}
"""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 


FIND_RELEVANT_CHUNK_SYSTEM_PROMPT = """
Determine whether or not the "Proposition" should belong to any of the existing chunks.

A proposition should belong to a chunk if their meaning, direction, or intentions are very similar. Be precise and strict in your judgement. 
The goal is to group only highly similar propositions and chunks provided in {source_language}.

If a proposition clearly fits within an existing chunk, return the "Chunk ID" without providing any explanations.
If the proposition does not clearly belong to an existing chunk, even if there is some overlap, return "No chunks" to indicate it should not be joined with any of the current chunks.

Example:
Input:
    - Proposition: "Greg really likes hamburgers"

    - Current Chunks:
        - Chunk ID: 2n4l3d
        - Chunk Name: Places in San Francisco
        - Chunk Summary: Overview of the things to do with San Francisco Places

        - Chunk ID: 93833k
        - Chunk Name: Food Greg likes
        - Chunk Summary: Lists of the food and dishes that Greg likes
Output: 93833k
"""


FIND_RELEVANT_CHUNK_USER_PROMPT = """
Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--
Determine if the following statement should belong to one of the chunks outlined in {source_language}:\n{proposition}\n
Important Note: For a given proposition, return only the 'Chunk ID' if there is a clear and strong similarity; otherwise, respond with 'No chunks'. Do not force propositions into existing chunks if they do not strongly fit. No explanations or additional text."
"""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 


NEW_CHUNK_TITLE_SYSTEM_PROMPT = """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
You should generate a very brief few word chunk title which will inform viewers what a chunk group is about in {source_language}.

A good chunk title is brief but encompasses what the chunk is about

You will be given a summary of a chunk which needs a title

Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else. Respond only with the new chunk title in {source_language}.
"""


NEW_CHUNK_TITLE_USER_PROMPT = """
Determine the title of the chunk that this summary belongs to:\n{summary}
"""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 


NEW_CHUNK_SUMMARY_SYSTEM_PROMPT = """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about in {source_language}.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the new chunk summary, nothing else. Respond only with the new chunk summary in {source_language}.
"""


NEW_CHUNK_SUMMARY_USER_PROMPT = """
Determine the summary of the new chunk that this proposition will go into:\n{proposition}
"""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 


UPDATE_CHUNK_TITLE_SYSTEM_PROMPT = """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about in {source_language}.

A good title will say what the chunk is about.

You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else. Respond only with the new chunk title in {source_language}.
"""


UPDATE_CHUNK_TITLE_USER_PROMPT = """
Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}
"""

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 


UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT = """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about in {source_language}.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a group of propositions which are in the chunk and the chunks current summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the chunk new summary, nothing else. Respond only with the chunk new summary in {source_language}.
"""


UPDATE_CHUNK_SUMMARY_USER_PROMPT = """
Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}
"""




