"""Curated seed corpus: legislation.gov.uk identifiers for each Act, and the
provision ids we keep from each (see plan: narrow curated slice, expandable later).

Section ids follow the pattern "section-<n>" (or "section-<n>-<sub>" for a cited
subsection). Schedule grounds are chunked at the cross-heading ("Pblock") level
with ids like "schedule-2-part-I-crossheading-ground-8".
"""

ACTS = [
    {
        "slug": "housing-act-1988",
        "path": "ukpga/1988/50",
        "expected_title": "Housing Act 1988",
        "wanted_ids": {
            "section-1", "section-5",
            "section-7",   # orders for possession - the operative "how can a landlord
                           # evict" provision now that s.21 is repealed
            "section-8", "section-9A", "section-13",
            "section-19A", "section-20", "section-21",
        },
        # All Schedule 2 grounds: the Renters' Rights Act 2025 reform added many new
        # ones (5A-5H, 6A/6B, 7B, 14ZA, 18 ...), and "which ground can they use"
        # is a core question, so take the whole list rather than hand-picking.
        "wanted_schedule_grounds": "all",
    },
    {
        "slug": "protection-from-eviction-act-1977",
        "path": "ukpga/1977/43",
        "expected_title": "Protection from Eviction Act 1977",
        "wanted_ids": {"section-1", "section-2", "section-3", "section-5"},
        "wanted_schedule_grounds": set(),
    },
    {
        "slug": "landlord-and-tenant-act-1985",
        "path": "ukpga/1985/70",
        "expected_title": "Landlord and Tenant Act 1985",
        "wanted_ids": {"section-11"},
        "wanted_schedule_grounds": set(),
    },
    {
        "slug": "homes-fitness-for-human-habitation-act-2018",
        "path": "ukpga/2018/34",
        "expected_title": "Homes (Fitness for Human Habitation) Act 2018",
        "wanted_ids": "all",  # short act, take every section
        "wanted_schedule_grounds": set(),
    },
    {
        "slug": "housing-act-2004",
        "path": "ukpga/2004/34",
        "expected_title": "Housing Act 2004",
        "wanted_ids": {
            "section-212", "section-213", "section-214", "section-215",
        },
        "wanted_schedule_grounds": set(),
        "wanted_schedules": {"schedule-10"},
    },
    {
        "slug": "tenant-fees-act-2019",
        "path": "ukpga/2019/4",
        "expected_title": "Tenant Fees Act 2019",
        "wanted_ids": "all",  # short act, take every section
        "wanted_schedule_grounds": set(),
    },
    {
        "slug": "renters-rights-act-2025",
        "path": "ukpga/2025/26",
        "expected_title": "Renters' Rights Act 2025",
        "wanted_ids": {
            "section-1",   # assured tenancies become periodic (no more fixed terms)
            "section-2",   # abolition of assured shorthold tenancies
            "section-3",   # changes to grounds for possession (replaces s.21 route)
            "section-4",   # possession for anti-social behaviour: relevant factors
            "section-5",   # form of notice of proceedings for possession
            "section-20",  # tenant notices to quit: timing (direct answer to "how much
                           # notice must I, the tenant, give" - amends PfEA 1977 s.5)
            "section-21",  # tenant notices to quit: other (new PfEA 1977 s.5A)
            "section-58",  # penalties for unlawful eviction/harassment
            "section-59",  # abandoned premises under former ASTs
            "section-60",  # remedying of hazards in dwelling-houses (England)
            "section-61",  # remedying of hazards in licensed accommodation
        },
        "wanted_schedule_grounds": set(),
    },
    # Statutory instruments (Docs.pdf §2 asks for "Acts and statutory instruments").
    # The prescribed FORM requirements live here, not in the parent Act: s.213(6)
    # Housing Act 2004 says information must be given "in the prescribed form"
    # without saying what it is, so questions about what a landlord actually has to
    # serve were unanswerable from primary legislation alone.
    {
        "slug": "tenancy-deposit-prescribed-information-order-2007",
        "path": "uksi/2007/797",
        "expected_title": "Housing (Tenancy Deposits) (Prescribed Information) Order 2007",
        "wanted_ids": "all",
        "wanted_schedule_grounds": set(),
    },
    # The Assured Tenancies (Forms) Regulations 2015 were tried and dropped: their
    # substance is the prescribed forms themselves, which live in Schedules this
    # pipeline doesn't ingest, so every regulation parsed to a bare heading. Five
    # empty chunks is not merely useless - heading-with-no-body is the signal the
    # repeal manifest uses, so they would have been reported to users as repealed
    # provisions. A parsing gap must not be able to masquerade as a repeal.
]

CASES = [
    {
        "slug": "superstrike-v-rodrigues-2013",
        "path": "ewca/civ/2013/669",
        "citation": "Superstrike Ltd v Rodrigues [2013] EWCA Civ 669",
    },
    {
        "slug": "charalambous-v-ng-2014",
        "path": "ewca/civ/2014/1604",
        "citation": "Charalambous v Ng [2014] EWCA Civ 1604",
    },
    {
        "slug": "johnson-v-old-2013",
        "path": "ewca/civ/2013/415",
        "citation": "Johnson v Old [2013] EWCA Civ 415",
    },
    {
        "slug": "spencer-v-taylor-2013",
        "path": "ewca/civ/2013/1600",
        "citation": "Taylor v Spencer [2013] EWCA Civ 1600",
    },
    {
        "slug": "trecarrell-house-v-rouncefield-2020",
        "path": "ewca/civ/2020/760",
        "citation": "Trecarrell House Ltd v Rouncefield [2020] EWCA Civ 760",
    },
    {
        "slug": "edwards-v-kumarasamy-2016",
        "path": "uksc/2016/40",
        "citation": "Edwards v Kumarasamy [2016] UKSC 40",
    },
]
