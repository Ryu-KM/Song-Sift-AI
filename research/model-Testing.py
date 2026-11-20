from sentence_transformers import SentenceTransformer

# Sentence Transformer model: https://huggingface.co/dunzhang/stella_en_400M_v5
model = SentenceTransformer('dunzhang/stella_en_400M_v5', trust_remote_code=True, config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

# Sentence Transformer model: https://huggingface.co/dunzhang/stella_en_1.5B_v5
# model = SentenceTransformer('dunzhang/stella_en_1.5B_v5', trust_remote_code=True, config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

# Sentence Transformer model: https://huggingface.co/jxm/cde-small-v1
# model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

# Sentence Transformer model: https://huggingface.co/nvidia/NV-Embed-v2
# Doesn't work :(
# model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

# Maximum number of tokens to be passed into model before being truncated
model.max_seq_length = 512
model.tokenizer.padding_side = "right"

sampleText = [
    {
        "query": "What are the impacts of artificial intelligence on job markets?",
        "content": [
            "AI is transforming industries by automating repetitive tasks, potentially displacing some jobs.",
            "Cloud computing enhances data storage and access for businesses.",
            "I am unemployed now because generative text models took my job."
        ]
    },
    {
        "query": "What are the effects of climate change on wildlife?",
        "content": [
            "Climate change disrupts habitats, leading to shifts in species distribution.",
            "Plastic pollution is a major threat to marine ecosystems.",
            "Many species are facing increased risk of extinction due to changing climates."
        ]
    },
    {
        "query": "What are the benefits of a plant-based diet?",
        "content": [
            "A plant-based diet can reduce the risk of chronic diseases and promote heart health.",
            "Regular hydration is essential for overall well-being.",
            "Incorporating more fruits and vegetables can improve digestive health."
        ]
    },
    {
        "query": "What are the must-see attractions in Paris?",
        "content": [
            "The Eiffel Tower is an iconic landmark and a must-visit in Paris.",
            "Local cuisine varies greatly across different regions.",
            "The Louvre Museum houses some of the world's most famous art pieces.",
            "dhsajhkghdjahgdiyuwdjhasdjhavsdjh abd dhasuihdiaughdaiu",
            "eiTflwre oTawrs an oicnac idmleankr an uMsvit-vi Parsi"
        ]
    },
    {
        "query": "What were the causes of World War I?",
        "content": [
            "The assassination of Archduke Franz Ferdinand triggered a series of political alliances.",
            "Economic competition among nations heightened tensions.",
            "Today I went to the store and purchased a can of Celsius fizz free peach mango."
        ]
    },
    {
        "query": "What are the health benefits of regular exercise?",
        "content": [
            "Regular exercise has numerous health benefits including improved cardiovascular health, weight management, and enhanced mental well-being.",
            "Eating fruits and vegetables is essential for a balanced diet.",
            "Exercise helps to strengthen muscles, improve flexibility, and increase stamina."
        ]
    },
    {
        "query": "I am a happy individual.",
        "content": [
            "My life is desolate and I am unhappy",
            "Eating fruits and vegetables is essential for a balanced diet.",
            "Each day that I go through life I find enjoyment and fulfillment."
        ]
    },
    {
        "query": "Love song",
        "content": [
            """
            We were both young when I first saw you
            I close my eyes and the flashback starts
            I'm standin' there
            On a balcony in summer air
            See the lights, see the party, the ball gowns
            See you make your way through the crowd
            And say, "Hello"
            Little did I know
            That you were Romeo, you were throwin' pebbles
            And my daddy said, "Stay away from Juliet"
            And I was cryin' on the staircase
            Beggin' you, "Please don't go, " and I said
            Romeo, take me somewhere we can be alone
            I'll be waiting, all there's left to do is run
            You'll be the prince and I'll be the princess
            It's a love story, baby, just say, "Yes"
            So I sneak out to the garden to see you
            We keep quiet, 'cause we're dead if they knew
            So close your eyes
            Escape this town for a little while, oh oh
            'Cause you were Romeo, I was a scarlet letter
            And my daddy said, "Stay away from Juliet"
            But you were everything to me
            I was beggin' you, "Please don't go, " and I said
            Romeo, take me somewhere we can be alone
            I'll be waiting, all there's left to do is run
            You'll be the prince and I'll be the princess
            It's a love story, baby, just say, "Yes"
            Romeo, save me, they're tryna tell me how to feel
            This love is difficult, but it's real
            Don't be afraid, we'll make it out of this mess
            It's a love story, baby, just say, "Yes"
            Oh, oh
            I got tired of waiting
            Wonderin' if you were ever comin' around
            My faith in you was fading
            When I met you on the outskirts of town, and I said
            Romeo, save me, I've been feeling so alone
            I keep waiting for you, but you never come
            Is this in my head? I don't know what to think
            He knelt to the ground and pulled out a ring
            And said, "Marry me, Juliet
            You'll never have to be alone
            I love you and that's all I really know
            I talked to your dad, go pick out a white dress
            It's a love story, baby, just say, "Yes"
            Oh, oh, oh
            Oh, oh, oh, oh
            'Cause we were both young when I first saw you
""",
            """You're talking shit for the hell of it
Addicted to betrayal, but you're relevant
You're terrified to look down
'Cause if you dare, you'll see the glare
Of everyone you burned just to get there
It's coming back around
And I keep my side of the street clean
You wouldn't know what I mean
'Cause karma is my boyfriend
Karma is a god
Karma is the breeze in my hair on the weekend
Karma's a relaxing thought
Aren't you envious that for you it's not?
Sweet like honey, karma is a cat
Purring in my lap 'cause it loves me
Flexing like a goddamn acrobat
Me and karma vibe like that
Spider-boy, king of thieves
Weave your little webs of opacity
My pennies made your crown
Trick me once, trick me twice
Don't you know that cash ain't the only price?
It's coming back around
And I keep my side of the street clean
You wouldn't know what I mean
'Cause karma is my boyfriend
Karma is a god
Karma is the breeze in my hair on the weekend
Karma's a relaxing thought
Aren't you envious that for you it's not?
Sweet like honey, karma is a cat
Purring in my lap 'cause it loves me
Flexing like a goddamn acrobat
Me and karma vibe like that
Ask me what I learned from all those years
Ask me what I earned from all those tears
Ask me why so many fade, but I'm still here
(I'm still, I'm still here)
'Cause karma is the thunder
Rattling your ground
Karma's on your scent like a bounty hunter
Karma's gonna track you down
Step by step from town to town
Sweet like justice, karma is a queen
Karma takes all my friends to the summit
Karma is the guy on the screen
Coming straight home to me
'Cause karma is my boyfriend (karma is my boyfriend)
Karma is a god
Karma is the breeze in my hair on the weekend (weekend)
Karma's a relaxing thought
Aren't you envious that for you it's not?
Sweet like honey, karma is a cat
Purring in my lap 'cause it loves me
Flexing like a goddamn acrobat
Me and karma vibe like that
Karma is my boyfriend
Karma is a god
Mm-hm
Karma's a relaxing thought
""",
            """I'm in the thick of it, everybody knows
They know me where it snows, I skied in and they froze
I don't know no nothin' 'bout no ice, I'm just cold
Forty somethin' milli' subs or so, I've been told

I'm in my prime, and this ain't even final form
They knocked me down, but still, my feet, they find the floor
I went from living rooms straight out to sold out tours
Life's a fight, but trust, I'm ready for the war

Woah-oh-oh
This is how the story goes
Woah-oh-oh
I guess this is how the story goes

I'm in the thick of it, everybody knows
They know me where it snows, I skied in and they froze
I don't know no nothin' 'bout no ice, I'm just cold
Forty somethin' milli' subs or so, I've been told

From the screen to the ring, to the pen, to the king
Where's my crown? That's my bling
Always drama when I ring
See, I believe that if I see it in my heart
Smash through the ceiling 'cause I'm reaching for the stars

Woah-oh-oh
This is how the story goes
Woah-oh-oh
I guess this is how the story goes

I'm in the thick of it, everybody knows
They know me where it snows, I skied in and they froze (Woo)
I don't know no nothin' 'bout no ice, I'm just cold
Forty somethin' milli' subs or so, I've been told

Highway to heaven, I'm just cruisin' by my lone'
They cast me out, left me for dead, them people cold
My faith in God, mind in the sun, I'm 'bout to sow (Yeah)
My life is hard, I took the wheel, I cracked the code (Yeah-yeah, woah-oh-oh)
Ain't nobody gon' save you, man, this life will break you (Yeah, woah-oh-oh)
In the thick of it, this is how the story goes

I'm in the thick of it, everybody knows
They know me where it snows, I skied in and they froze
I don't know no nothin' 'bout no ice, I'm just cold
Forty somethin' milli' subs or so, I've been told
I'm in the thick of it, everybody knows (Everybody knows)
They know me where it snows, I skied in and they froze (Yeah)
I don't know no nothin' 'bout no ice, I'm just cold
Forty somethin' milli' subs or so, I've been told (Ooh-ooh)

Woah-oh-oh (Nah-nah-nah-nah, ayy, ayy)
This is how the story goes (Nah, nah)
Woah-oh-oh
I guess this is how the story goes
"""
        ]
    },
]

def add_eos(input_examples):
    eos_token = model.tokenizer.eos_token
    if eos_token is not None:
        input_examples = [input_example + eos_token for input_example in input_examples]
    return input_examples

batch_size = 2

for sample in sampleText:
    query = sample["query"]
    content = sample["content"]

    # dot product between query and content embeddings
    query_embedding = model.encode(add_eos([query]), batch_size=batch_size, normalize_embeddings=True)
    passage_embeddings = model.encode(add_eos(content), batch_size=batch_size, normalize_embeddings=True)

    scores = (query_embedding @ passage_embeddings.T) * 100

    print(f"\nQuery: {query}")
    print("Similarity Scores:", scores.tolist())
