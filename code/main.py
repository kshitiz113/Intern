"""
MLE Hiring Challenge — Support Triage Agent
Main entry point. Reads support_tickets.csv, processes each ticket, writes output.csv.

Usage:
    python main.py
    python main.py --input path/to/input.csv --output path/to/output.csv
    python main.py --verbose
"""

import csv
import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path

# Add code/ to sys.path so imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import Ticket, TicketResult
from llm_client import LLMClient
from corpus_indexer import CorpusIndexer
from retriever import HybridRetriever
from safety import SafetyShield
from agent import TriageAgent

# ── Logging Setup ─────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ── CSV I/O ───────────────────────────────────────────────────────────

def read_tickets(input_path: str) -> list:
    """Read support tickets from CSV."""
    tickets = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            ticket = Ticket(
                issue_raw=row.get("Issue", row.get("issue", "")),
                subject=row.get("Subject", row.get("subject", "")),
                company=row.get("Company", row.get("company", "")),
                row_index=i,
            )
            tickets.append(ticket)
    return tickets


def write_results(results: list, output_path: str):
    """Write results to output CSV with all 14 required columns."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=config.OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in results:
            row = {
                "issue": result.issue,
                "subject": result.subject,
                "company": result.company,
                "response": result.response,
                "product_area": result.product_area,
                "status": result.status,
                "request_type": result.request_type,
                "justification": result.justification,
                "confidence_score": str(result.confidence_score),
                "source_documents": result.source_documents,
                "risk_level": result.risk_level,
                "pii_detected": result.pii_detected,
                "language": result.language,
                "actions_taken": result.actions_taken,
            }
            writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--input", default=str(config.INPUT_CSV), help="Input CSV path")
    parser.add_argument("--output", default=str(config.OUTPUT_CSV), help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("MLE Hiring Challenge — Support Triage Agent")
    logger.info("=" * 60)

    # ── 1. Initialize LLM client ──────────────────────────────────
    logger.info("Initializing LLM client...")
    llm = LLMClient()

    # ── 2. Build corpus index ─────────────────────────────────────
    logger.info("Building corpus index (BM25 + embeddings)...")
    index_start = time.time()
    indexer = CorpusIndexer(llm_client=llm)
    indexer.build_index()
    index_time = time.time() - index_start
    logger.info(f"Corpus indexed: {indexer.size} documents in {index_time:.1f}s")

    # ── 3. Initialize retriever + safety shield ───────────────────
    retriever = HybridRetriever(indexer, llm_client=llm)
    safety = SafetyShield(llm_client=llm, corpus_indexer=indexer)

    # ── 4. Create agent ───────────────────────────────────────────
    agent = TriageAgent(
        llm_client=llm,
        indexer=indexer,
        retriever=retriever,
        safety_shield=safety,
    )

    # ── 5. Read input tickets ─────────────────────────────────────
    logger.info(f"Reading tickets from {args.input}...")
    tickets = read_tickets(args.input)
    logger.info(f"Loaded {len(tickets)} tickets")

    # ── 6. Process tickets ────────────────────────────────────────
    results = []
    total = len(tickets)

    for i, ticket in enumerate(tickets):
        ticket_start = time.time()
        try:
            result = agent.process_ticket(ticket)
        except Exception as e:
            logger.error(f"CRITICAL: Ticket {i+1} crashed: {e}")
            result = TicketResult(
                issue=ticket.issue_raw,
                subject=ticket.subject,
                company=ticket.company,
                status="escalated",
                product_area="general",
                request_type="product_issue",
                response="I apologize, but I was unable to process your request. A human agent will assist you shortly.",
                justification="Agent processing error — escalated for safety.",
                confidence_score=0.2,
                risk_level="medium",
                pii_detected="false",
                language="en",
                actions_taken=json.dumps([{
                    "action": "escalate_to_human",
                    "parameters": {"priority": "normal", "department": "general", "summary": "Processing error"},
                }]),
            )
        results.append(result)
        ticket_time = time.time() - ticket_start

        # Progress logging
        elapsed = time.time() - start_time
        logger.info(
            f"[{i+1}/{total}] {result.status:>9s} | {result.request_type:>15s} | "
            f"conf={result.confidence_score:.2f} | risk={result.risk_level:>8s} | "
            f"{ticket_time:.1f}s | total={elapsed:.0f}s"
        )

    # ── 7. Write output ───────────────────────────────────────────
    logger.info(f"Writing results to {args.output}...")
    write_results(results, args.output)

    # ── 8. Summary ────────────────────────────────────────────────
    total_time = time.time() - start_time
    replied = sum(1 for r in results if r.status == "replied")
    escalated = sum(1 for r in results if r.status == "escalated")
    avg_conf = sum(r.confidence_score for r in results) / max(len(results), 1)
    pii_count = sum(1 for r in results if r.pii_detected == "true")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Total tickets:     {total}")
    logger.info(f"  Replied:           {replied}")
    logger.info(f"  Escalated:         {escalated}")
    logger.info(f"  PII detected:      {pii_count}")
    logger.info(f"  Avg confidence:    {avg_conf:.2f}")
    logger.info(f"  Total time:        {total_time:.1f}s")
    logger.info(f"  LLM calls:         {llm.call_count}")
    logger.info(f"  Output:            {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
