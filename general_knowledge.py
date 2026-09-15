def seed_geography_knowledge(brain):
    capitals = [
        ("india", "new_delhi"), ("usa", "washington_dc"), ("uk", "london"),
        ("france", "paris"), ("germany", "berlin"), ("japan", "tokyo"),
        ("china", "beijing"), ("russia", "moscow"), ("australia", "canberra"),
        ("brazil", "brasilia"), ("canada", "ottawa"), ("italy", "rome"),
        ("spain", "madrid"), ("egypt", "cairo"), ("pakistan", "islamabad"),
        ("nepal", "kathmandu"), ("bangladesh", "dhaka"), ("sri_lanka", "colombo"),
        ("south_korea", "seoul"), ("mexico", "mexico_city"),
    ]
    for country, capital in capitals:
        brain.add_fact("capital_of", country, capital)
        brain.add_relation(country, "is_a", "country")

    continents = [
        ("india", "asia"), ("china", "asia"), ("japan", "asia"),
        ("pakistan", "asia"), ("nepal", "asia"), ("bangladesh", "asia"),
        ("sri_lanka", "asia"), ("south_korea", "asia"),
        ("usa", "north_america"), ("canada", "north_america"), ("mexico", "north_america"),
        ("uk", "europe"), ("france", "europe"), ("germany", "europe"),
        ("italy", "europe"), ("spain", "europe"), ("russia", "europe"),
        ("brazil", "south_america"),
        ("egypt", "africa"),
        ("australia", "oceania"),
    ]
    for country, continent in continents:
        brain.add_fact("located_in", country, continent)
        brain.add_relation(continent, "is_a", "continent")

    largest_by_population = ["china", "india", "usa", "indonesia", "pakistan"]
    for rank, country in enumerate(largest_by_population, start=1):
        brain.add_fact("population_rank", country, str(rank))

    famous_rivers = [
        ("nile", "egypt"), ("ganges", "india"), ("amazon", "brazil"),
        ("thames", "uk"), ("seine", "france"), ("yangtze", "china"),
    ]
    for river, country in famous_rivers:
        brain.add_fact("river_in", river, country)
        brain.add_relation(river, "is_a", "river")


def seed_solar_system_knowledge(brain):
    planets_in_order = [
        "mercury", "venus", "earth", "mars", "jupiter",
        "saturn", "uranus", "neptune",
    ]
    for position, planet in enumerate(planets_in_order, start=1):
        brain.add_fact("orbit_position", planet, str(position))
        brain.add_fact("orbits", planet, "sun")
        brain.add_relation(planet, "is_a", "planet")

    brain.add_fact("largest_planet", "jupiter")
    brain.add_fact("smallest_planet", "mercury")
    brain.add_fact("closest_to_sun", "mercury")
    brain.add_fact("farthest_from_sun", "neptune")
    brain.add_fact("has_rings", "saturn")
    brain.add_fact("has_rings", "jupiter")
    brain.add_fact("has_rings", "uranus")
    brain.add_fact("has_rings", "neptune")
    brain.add_fact("known_for", "earth", "supports_life")
    brain.add_fact("known_for", "mars", "red_color")
    brain.add_fact("known_for", "venus", "hottest_planet")

    brain.add_relation("sun", "is_a", "star")
    brain.add_relation("moon", "is_a", "natural_satellite")
    brain.add_fact("orbits", "moon", "earth")


def seed_basic_science_knowledge(brain):
    elements = [
        ("hydrogen", "H", "1"), ("helium", "He", "2"), ("carbon", "C", "6"),
        ("oxygen", "O", "8"), ("nitrogen", "N", "7"), ("iron", "Fe", "26"),
        ("gold", "Au", "79"), ("sodium", "Na", "11"),
    ]
    for name, symbol, atomic_number in elements:
        brain.add_fact("element_symbol", name, symbol)
        brain.add_fact("atomic_number", name, atomic_number)
        brain.add_relation(name, "is_a", "chemical_element")

    brain.add_fact("boiling_point_celsius", "water", "100")
    brain.add_fact("freezing_point_celsius", "water", "0")
    brain.add_fact("chemical_formula", "water", "H2O")
    brain.add_fact("chemical_formula", "carbon_dioxide", "CO2")
    brain.add_fact("chemical_formula", "table_salt", "NaCl")

    brain.add_fact("speed_of_light_km_per_second", "light", "299792")
    brain.add_fact("gravity_discoverer", "gravity", "isaac_newton")
    brain.add_fact("relativity_discoverer", "relativity", "albert_einstein")
    brain.add_fact("evolution_theorist", "evolution", "charles_darwin")

    animal_classes = [
        ("dog", "mammal"), ("cat", "mammal"), ("elephant", "mammal"), ("human", "mammal"),
        ("eagle", "bird"), ("sparrow", "bird"), ("penguin", "bird"),
        ("snake", "reptile"), ("crocodile", "reptile"), ("lizard", "reptile"),
        ("salmon", "fish"), ("shark", "fish"),
        ("frog", "amphibian"),
    ]
    for animal, animal_class in animal_classes:
        brain.add_relation(animal, "is_a", animal_class)
    for animal_class in {"mammal", "bird", "reptile", "fish", "amphibian"}:
        brain.add_relation(animal_class, "is_a", "animal")
    brain.add_relation("animal", "is_a", "living_being")
    brain.add_relation("plant", "is_a", "living_being")


def seed_history_knowledge(brain):
    events = [
        ("world_war_2", "1939", "1945"),
        ("world_war_1", "1914", "1918"),
        ("indian_independence", "1947", "1947"),
        ("moon_landing", "1969", "1969"),
        ("french_revolution", "1789", "1799"),
    ]
    for event, start_year, end_year in events:
        brain.add_fact("started_in", event, start_year)
        brain.add_fact("ended_in", event, end_year)

    brain.add_fact("first_president_of", "usa", "george_washington")
    brain.add_fact("first_pm_of", "india", "jawaharlal_nehru")
    brain.add_fact("wrote", "theory_of_relativity", "albert_einstein")
    brain.add_fact("wrote", "origin_of_species", "charles_darwin")
    brain.add_fact("invented", "telephone", "alexander_graham_bell")
    brain.add_fact("invented", "light_bulb", "thomas_edison")
    brain.add_fact("invented", "airplane", "wright_brothers")


def seed_all_general_knowledge(brain):
    seed_geography_knowledge(brain)
    seed_solar_system_knowledge(brain)
    seed_basic_science_knowledge(brain)
    seed_history_knowledge(brain)
    brain.knowledge_graph.mark_transitive("is_a")
    brain.run_forward_chaining()
    return brain.fact_store.size()


def register_general_knowledge_intent(brain):
    brain.register_intent_examples("general_knowledge_query", [
        "what is the capital of india",
        "what is the capital of france",
        "how many planets are there",
        "what is the chemical formula of water",
        "who discovered gravity",
        "who invented the telephone",
        "what is the largest planet",
        "when did world war 2 start",
        "what is the atomic number of oxygen",
        "which river is in egypt",
    ])

    def general_knowledge_handler(text, engine_ref, parsed_sentences):
        lowered = text.lower()
        words = lowered.replace("?", "").split()

        if "river" in words:
            for river_fact in engine_ref.fact_store.all_by_predicate("river_in"):
                if river_fact.args[1] in words:
                    return f"The {river_fact.args[0].capitalize()} river flows through {river_fact.args[1].replace('_', ' ')}."

        if "capital" in words:
            for country_fact in engine_ref.fact_store.all_by_predicate("capital_of"):
                if country_fact.args[0] in words:
                    return f"The capital of {country_fact.args[0].replace('_', ' ')} is {country_fact.args[1].replace('_', ' ')}."

        if "planets" in words or "planet" in words:
            if "largest" in words:
                return "Jupiter is the largest planet in the solar system."
            if "smallest" in words:
                return "Mercury is the smallest planet in the solar system."
            if "how" in words and "many" in words:
                return "There are eight planets in the solar system."

        if "formula" in words:
            for formula_fact in engine_ref.fact_store.all_by_predicate("chemical_formula"):
                subject = formula_fact.args[0].replace("_", " ")
                if subject in lowered:
                    return f"The chemical formula of {subject} is {formula_fact.args[1]}."

        if "atomic" in words or "number" in words:
            for element_fact in engine_ref.fact_store.all_by_predicate("atomic_number"):
                if element_fact.args[0] in words:
                    return f"The atomic number of {element_fact.args[0]} is {element_fact.args[1]}."

        if "gravity" in lowered:
            return "Isaac Newton is credited with discovering gravity."
        if "telephone" in lowered:
            return "Alexander Graham Bell invented the telephone."
        if "light bulb" in lowered or "lightbulb" in lowered:
            return "Thomas Edison is credited with inventing the light bulb."
        if "world war 2" in lowered or "world war ii" in lowered:
            return "World War 2 started in 1939 and ended in 1945."

        return "I have some general knowledge, but I could not find a direct answer for that."

    brain.register_intent_handler("general_knowledge_query", general_knowledge_handler)
