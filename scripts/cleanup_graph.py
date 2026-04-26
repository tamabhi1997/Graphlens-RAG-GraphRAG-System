"""
Neo4j Concept Deduplication Cleanup Script
==========================================
Run this ONCE to clean up existing duplicate concept nodes in Neo4j.
After running this, entity_normaliser.py prevents new duplicates going forward.

Usage:
    python scripts/cleanup_graph.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphlens.graphrag.entity_normaliser import normalise_concept
from neo4j import GraphDatabase


def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
    )


def fetch_all_concepts(session) -> list[str]:
    result = session.run("MATCH (c:Concept) RETURN c.name AS name")
    return [record["name"] for record in result]


def merge_duplicate(session, old_name: str, canonical_name: str) -> int:
    """Rewire all edges from old_name to canonical_name, then delete old node."""

    # Ensure canonical node exists
    session.run(
        "MERGE (c:Concept {name: $name}) ON CREATE SET c.mention_count = 0",
        name=canonical_name
    )

    # Rewire MENTIONS edges (Chunk → Concept)
    result = session.run(
        """
        MATCH (chunk:Chunk)-[r:MENTIONS]->(old:Concept {name: $old})
        MATCH (canonical:Concept {name: $canonical})
        WHERE old <> canonical
        MERGE (chunk)-[:MENTIONS]->(canonical)
        DELETE r
        RETURN count(r) AS rewired
        """,
        old=old_name, canonical=canonical_name
    )
    record = result.single()
    rewired = record["rewired"] if record else 0

    # Rewire outgoing concept relationships
    session.run(
        """
        MATCH (old:Concept {name: $old})-[r]->(target:Concept)
        MATCH (canonical:Concept {name: $canonical})
        WHERE old <> canonical AND target <> old
        MERGE (canonical)-[:RELATES_TO]->(target)
        DELETE r
        """,
        old=old_name, canonical=canonical_name
    )

    # Rewire incoming concept relationships
    session.run(
        """
        MATCH (source:Concept)-[r]->(old:Concept {name: $old})
        MATCH (canonical:Concept {name: $canonical})
        WHERE old <> canonical AND source <> old
        MERGE (source)-[:RELATES_TO]->(canonical)
        DELETE r
        """,
        old=old_name, canonical=canonical_name
    )

    # Delete old node if it has no more relationships
    session.run(
        """
        MATCH (old:Concept {name: $old})
        WHERE NOT (old)--()
        DELETE old
        """,
        old=old_name
    )

    return rewired


def update_mention_counts(session):
    session.run(
        """
        MATCH (c:Concept)
        OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(c)
        WITH c, count(chunk) AS cnt
        SET c.mention_count = cnt
        """
    )


def run_cleanup():
    driver = get_driver()

    print("=" * 60)
    print("GraphLens — Neo4j Concept Deduplication Cleanup")
    print("=" * 60)

    with driver.session() as session:

        # Fetch all concept names
        raw_names = fetch_all_concepts(session)
        print(f"\nFound {len(raw_names)} concept nodes")

        # Group by normalised name
        groups: dict[str, list[str]] = defaultdict(list)
        skipped = []

        for name in raw_names:
            canonical = normalise_concept(name)
            if canonical:
                groups[canonical].append(name)
            else:
                skipped.append(name)

        duplicates = {k: v for k, v in groups.items() if len(v) > 1}
        clean      = {k: v for k, v in groups.items() if len(v) == 1}

        print(f"\nAfter normalisation:")
        print(f"  Clean (unique):          {len(clean)}")
        print(f"  Groups with duplicates:  {len(duplicates)}")
        print(f"  Skipped (noise):         {len(skipped)}")

        if not duplicates:
            print("\nNo duplicates found — graph is already clean!")
            driver.close()
            return

        print(f"\nMerging {len(duplicates)} duplicate groups...")
        print("-" * 60)

        total_merged = 0
        total_edges  = 0

        for canonical_name, raw in duplicates.items():
            others = [n for n in raw if n != canonical_name]
            if not others:
                # all names are the same as canonical — nothing to merge
                continue

            print(f"\n  '{canonical_name}' ← merging {len(others)} variant(s):")
            for old in others:
                print(f"    - '{old}'")
                edges = merge_duplicate(session, old, canonical_name)
                total_edges  += edges
                total_merged += 1

        # Recalculate mention counts
        print("\nRecalculating mention counts...")
        update_mention_counts(session)

        # Final count
        result = session.run("MATCH (c:Concept) RETURN count(c) AS total")
        final_count = result.single()["total"]

        print("\n" + "=" * 60)
        print("Cleanup complete!")
        print(f"  Nodes merged:         {total_merged}")
        print(f"  Edges rewired:        {total_edges}")
        print(f"  Final concept count:  {final_count}")
        print("=" * 60)
        print("\nGraph expansion should now fire more reliably.")
        print("Re-run the ablation study to see improved GraphRAG metrics.")

    driver.close()


if __name__ == "__main__":
    run_cleanup()