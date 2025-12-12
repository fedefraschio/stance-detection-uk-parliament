import json
import os 

records = {
    "nuclear":{
        'keywords':[
            "Power plant",
            "Nuclear power generation",
            "Decommissioning",
            "Nuclear fuel",
            "Radioactivity",
            "Radioactive waste",
            "Energy base load power source",
            "Fukushima",
            "Chernobyl",
            "Daiichi Nuclear Power Plant",
            # General nuclear-energy vocabulary
            "Nuclear reactors",
            "Pressurised Water Reactor",
            "Advanced Gas-cooled Reactor",
            "Small Modular Reactors",
            "Advanced Modular Reactors",
            "Nuclear fission",
            "Nuclear fusion",
            "Uranium enrichment",
            "Fuel reprocessing",
            "Spent fuel management",
            "Nuclear contamination",
            "Nuclear decontamination",
            "Radiation leakage",
            # UK-specific nuclear infrastructure
            "Hinkley Point C",
            "Sizewell C",
            "Wylfa Newydd",
            "Oldbury",
            "Bradwell B",
            "Sellafield",
            "Dounreay",
            "Magnox reactors",
            "Office for Nuclear Regulation",
            "Nuclear Decommissioning Authority",
            "Energy Security Strategy",
            "Regulated Asset Base model",
            "RAB model",
            # Political and policy themes
            "Nuclear financing",
            # Risk and crisis terms
            "Nuclear meltdown",
            "Core containment",
            "Emergency cooling system",
            # Geopolitical and supply chain
            "Uranium supply chain",
            "Nuclear cooperation agreements",
            "Russian uranium",
            ],
        'policyarea':[7, 8],
        
        },

    "climate change": {
        'keywords':
            ["global warming",
             "greenhouse gases", "carbon emissions", "renewable energy", "sustainability",  
             "climate mitigation", "COP26", "Paris Agreement"],
        'policyarea':[8],
        
        },
    "Gaza":{ 
        'keywords': [
            # Israeli political figures
            "Netanyahu",
            "Benjamin Netanyahu",
            "Ehud Olmert",
            "Ariel Sharon",
            "Ehud Barak",
            "Tzipi Livni",
            "Naftali Bennett",
            "Yair Lapid",
            "Avigdor Lieberman",
            "Shimon Peres",
            "Isaac Herzog",
            "Reuven Rivlin",
            "Benny Gantz",
            "Yoav Gallant",
            "Ben Gvir",
            "Zionism", "Zionist", "anti-zionist", "anti-zionism",

            # Palestinian political figures
            "Mahmoud Abbas",
            "Abu Mazen",
            "Ismail Haniyeh",
            "Yahya Sinwar",
            "Marwan Barghouti",
            "Saeb Erekat",
            "Mohammad Shtayyeh",
            "Khaled Mashal",
            "Ahmed Yassin",
            "Mohammed Deif",

            # Israeli institutions / organisations
            "Knesset",

            "Israel Defense Forces",
            "Shin Bet",
            "Mossad",
            "Likud",
            "Yesh Atid",
            "Blue and White",
            "Kadima",
            "Religious Zionism party",
            "Shas",
            "Meretz",

            # Palestinian institutions / organisations
            "Palestine Liberation Organization",
            "Fatah",
            "Hamas",
            "Popular Front for the Liberation of Palestine",
            "PFLP",
            "Al-Qassam Brigades",
            "Palestinian Legislative Council",

            # Key places (Gaza / Israel)
            "Gaza City",
            "Rafah",
            "Khan Younis",
            "Jabalia",
            "Beit Hanoun",
            "Deir al-Balah",
            "Shuja'iyya",
            "Nuseirat",
            "Beit Lahia",
            "Al-Shifa Hospital",

            # Key places (Israel)
            "Sderot",
            "Ashkelon",
            "Ashdod",
            "Beersheba",
            "Tel Aviv",
            "Jerusalem",
            "West Jerusalem",
            "East Jerusalem",
            "Haifa",
            "Eilat",

            # Crossings and checkpoints
            "Erez Crossing",
            "Kerem Shalom Crossing",
            "Rafah Crossing",
            "Qalandia",
            "Allenby Bridge Crossing",
            "Gilo checkpoint",

            # Named military operations (appear frequently in UK Parliament debates)
            "Operation Cast Lead",
            "Operation Pillar of Defense",
            "Operation Protective Edge",
            "Operation Guardian of the Walls",
            "Operation Breaking Dawn",
            "Operation Iron Swords",
            "Operation Summer Rains",
            "Operation Autumn Clouds",

            # International actors mentioned regarding the conflict
            "UNRWA",
            "UN Human Rights Council",
            "Quartet on the Middle East",
            

            # Other region-specific named entities
            "Al-Aqsa Mosque",
            "Temple Mount",
            "Haram al-Sharif",
            "Sheikh Jarrah",
            "Hebron",
            "Nablus",
            "Ramallah",
            "Bethlehem",
            "Jenin",

            # NGOs / rights groups mentioned often
            "B'Tselem",
            "Breaking the Silence",
            "ICRC"
        ],
        'policyarea':[16, 18, 19],
        
    }}

path = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', 'records.json')

with open(path, 'w') as f:
    json.dump(records, f, indent=4)

