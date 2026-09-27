"""
Submission Validator (stdlib only — no external dependencies)
=============================================================
Checks matching_results.tsv and candidate_pairs.tsv against every
contest rule before you spend a submission on it.

Usage (run from the project root):
  python utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test

Prints PASS (exit 0) when safe to submit, or numbered issues (exit 1).
"""

import argparse
import os
import sys
import time


def load_s1_ids(test_dir):
    """Load test S1 IDs efficiently."""
    s1_ids = set()
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(s1_path):
        for fname in os.listdir(test_dir):
            if "source1" in fname and fname.endswith(".tsv"):
                s1_path = os.path.join(test_dir, fname)
                break
    if not os.path.isfile(s1_path):
        return s1_ids

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            parts = line.split("\t")
            if parts:
                eid = parts[0].strip()
                if eid:
                    s1_ids.add(eid)
    return s1_ids


def load_s23_ids(test_dir):
    """Load test S2 and S3 IDs into a set."""
    s23_ids = set()
    for fname in sorted(os.listdir(test_dir)):
        if ("source2" in fname or "source3" in fname) and fname.endswith(".tsv"):
            fpath = os.path.join(test_dir, fname)
            print(f"  Indexing {fname}...")
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline()
                for line in f:
                    parts = line.split("\t")
                    if parts:
                        eid = parts[0].strip()
                        if eid:
                            s23_ids.add(eid)
    return s23_ids


def validate(matching_path, candidate_path, test_dir):
    issues = []

    if not os.path.isdir(test_dir):
        issues.append(f"Test directory not found: {test_dir}")
        return issues

    print("Step 1: Loading test S1 entity IDs...")
    t0 = time.time()
    s1_ids = load_s1_ids(test_dir)
    print(f"  Loaded {len(s1_ids):,} test S1 IDs in {time.time()-t0:.1f}s")
    if not s1_ids:
        issues.append("No S1 entity IDs found in test directory")
        return issues

    print("Step 2: Loading test S2/S3 entity IDs...")
    t0 = time.time()
    s23_ids = load_s23_ids(test_dir)
    print(f"  Loaded {len(s23_ids):,} test S2/S3 IDs in {time.time()-t0:.1f}s")
    if not s23_ids:
        issues.append("No S2/S3 entity IDs found in test directory")
        return issues

    if not os.path.isfile(matching_path):
        issues.append(f"matching_results.tsv not found: {matching_path}")
        return issues

    if not os.path.isfile(candidate_path):
        issues.append(f"candidate_pairs.tsv not found: {candidate_path}")
        return issues

    print("Step 3: Streaming and validating matching_results and candidate_pairs...")
    t0 = time.time()
    seen_s1_match = set()
    seen_s1_cand = set()
    total_matches = 0
    total_candidates = 0
    total_singletons = 0
    line_num = 1

    with open(matching_path, "r", encoding="utf-8", errors="replace") as fm, \
         open(candidate_path, "r", encoding="utf-8", errors="replace") as fc:

        # Header check
        m_head = fm.readline().strip().split("\t")
        c_head = fc.readline().strip().split("\t")

        if len(m_head) < 2 or m_head[0] != "source1_entity_id" or m_head[1] != "matched_entity_ids":
            issues.append(f"matching_results.tsv invalid header: {m_head}")
        if len(c_head) < 2 or c_head[0] != "source1_entity_id" or c_head[1] != "candidate_entity_ids":
            issues.append(f"candidate_pairs.tsv invalid header: {c_head}")

        if issues:
            return issues

        for m_line, c_line in zip(fm, fc):
            line_num += 1
            if line_num % 500_000 == 0:
                print(f"  Verified {line_num-1:,} rows...")

            # Parse matching line
            m_parts = m_line.rstrip("\r\n").split("\t")
            m_s1 = m_parts[0].strip() if len(m_parts) > 0 else ""
            m_matched_str = m_parts[1].strip() if len(m_parts) > 1 else ""

            # Parse candidate line
            c_parts = c_line.rstrip("\r\n").split("\t")
            c_s1 = c_parts[0].strip() if len(c_parts) > 0 else ""
            c_cand_str = c_parts[1].strip() if len(c_parts) > 1 else ""

            # Row alignment
            if m_s1 != c_s1:
                if len(issues) < 20:
                    issues.append(f"Row {line_num} S1 ID mismatch: matching has '{m_s1}', candidate has '{c_s1}'")

            # Duplicate S1 check
            if m_s1 in seen_s1_match:
                if len(issues) < 20:
                    issues.append(f"matching_results.tsv row {line_num}: duplicate source1_entity_id '{m_s1}'")
            seen_s1_match.add(m_s1)

            if c_s1 in seen_s1_cand:
                if len(issues) < 20:
                    issues.append(f"candidate_pairs.tsv row {line_num}: duplicate source1_entity_id '{c_s1}'")
            seen_s1_cand.add(c_s1)

            # Valid S1 ID
            if m_s1 not in s1_ids:
                if len(issues) < 20:
                    issues.append(f"matching_results.tsv row {line_num}: '{m_s1}' not in test S1 set")

            # Parse candidate set
            cand_set = set()
            if c_cand_str:
                cand_list = [c.strip() for c in c_cand_str.split(",") if c.strip()]
                if len(cand_list) != len(set(cand_list)):
                    if len(issues) < 20:
                        issues.append(f"candidate_pairs.tsv row {line_num} ({c_s1}): duplicate candidate IDs")
                cand_set = set(cand_list)
                total_candidates += len(cand_set)

            # Parse match list
            if m_matched_str:
                match_list = [m.strip() for m in m_matched_str.split(",") if m.strip()]
                if len(match_list) != len(set(match_list)):
                    if len(issues) < 20:
                        issues.append(f"matching_results.tsv row {line_num} ({m_s1}): duplicate matched IDs")

                total_matches += len(match_list)
                for mid in match_list:
                    if mid.startswith("S1-"):
                        if len(issues) < 20:
                            issues.append(f"matching_results.tsv row {line_num} ({m_s1}): self-match/S1 ID in matches '{mid}'")
                    elif mid not in s23_ids:
                        if len(issues) < 20:
                            issues.append(f"matching_results.tsv row {line_num} ({m_s1}): matched ID '{mid}' not in test S2/S3 IDs")

                    # Crucial constraint: matched ID MUST be in candidates
                    if mid not in cand_set:
                        if len(issues) < 20:
                            issues.append(f"Pipeline bug row {line_num} ({m_s1}): matched ID '{mid}' not in candidate list")
            else:
                total_singletons += 1

        # Check for trailing lines in either file
        m_extra = fm.readline()
        c_extra = fc.readline()
        if m_extra:
            issues.append("matching_results.tsv has extra rows compared to candidate_pairs.tsv")
        if c_extra:
            issues.append("candidate_pairs.tsv has extra rows compared to matching_results.tsv")

    # Completeness check: every S1 entity must appear
    missing_in_match = s1_ids - seen_s1_match
    if missing_in_match:
        sample = sorted(missing_in_match)[:5]
        issues.append(f"matching_results.tsv: {len(missing_in_match)} test S1 entities missing! Sample: {sample}")

    missing_in_cand = s1_ids - seen_s1_cand
    if missing_in_cand:
        sample = sorted(missing_in_cand)[:5]
        issues.append(f"candidate_pairs.tsv: {len(missing_in_cand)} test S1 entities missing! Sample: {sample}")

    print(f"  Validation finished in {time.time()-t0:.1f}s")
    print(f"\nSummary Statistics:")
    print(f"  Total test S1 entities : {len(s1_ids):,}")
    print(f"  Total matched entities : {len(seen_s1_match) - total_singletons:,}")
    print(f"  Total singleton entities: {total_singletons:,} ({total_singletons/len(s1_ids)*100:.2f}%)")
    print(f"  Total candidate pairs  : {total_candidates:,} (avg {total_candidates/len(s1_ids):.1f}/entity)")
    print(f"  Total matches predicted: {total_matches:,} (avg {total_matches/len(s1_ids):.2f}/entity)")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission files before uploading."
    )
    parser.add_argument(
        "--matching",
        default="output/matching_results.tsv",
        help="Path to matching_results.tsv",
    )
    parser.add_argument(
        "--candidate",
        default="output/candidate_pairs.tsv",
        help="Path to candidate_pairs.tsv",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Directory with test source TSV files",
    )
    args = parser.parse_args()

    print(f"Validating submission...")
    print(f"  matching : {args.matching}")
    print(f"  candidate: {args.candidate}")
    print(f"  test-dir : {args.test_dir}")
    print()

    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print("\nPASS -- submission files are completely valid and safe to upload.")
        sys.exit(0)
    else:
        print(f"\nFAIL -- {len(issues)} issue(s) found:\n")
        for idx, issue in enumerate(issues, start=1):
            print(f"  {idx}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
