from __future__ import annotations
import csv
import json
import toml
import queue
import threading
import sys
import os
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, Sequence, List, TYPE_CHECKING
from tqdm.auto import tqdm

from src.analysis.analyst_module import Analyst
from src.analysis.supabase_uploader import SupabaseUploader

class BatchRunner:
    """
    Handles batch processing of earnings call transcripts using an Analyst instance.
    Manages file iteration, threading (concurrent I/O), progress bars, and CSV summarization.
    """
    def __init__(self, analyst: "Analyst"):
        self.analyst = analyst

    def _resolve_num_threads(self, requested_threads: Optional[int]) -> int:
        """
        Determines the effective number of threads to use.
        Checks if GIL is enabled/disabled and limits threads if running in a free-threaded environment.
        """
        if not requested_threads or requested_threads < 1:
            return 1

        # Check if GIL is enabled
        gil_enabled = True
        if hasattr(sys, "_is_gil_enabled"):
            gil_enabled = sys._is_gil_enabled()
        elif hasattr(sys.flags, "nogil"):
            gil_enabled = not sys.flags.nogil

        # If GIL is disabled (free-threaded), ensure we don't exceed system threads
        if not gil_enabled:
            max_threads = os.cpu_count() or 1
            if requested_threads > max_threads:
                return max_threads
        
        return requested_threads

    @staticmethod
    def _flatten_analysis_toml(toml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten the analysis_metadata.toml structure into a single dict suitable for CSV.
        Keeps stable column names across files.
        """
        doc = toml_data.get("document", {})
        meta = toml_data.get("metadata_matching", {})
        da = toml_data.get("document_attr", {})

        start_date_str = doc.get("start_date")
        date_val, time_val = None, None
        if start_date_str:
            try:
                # The format is set in Analyst
                dt = datetime.strptime(start_date_str, "%d-%b-%y %I:%M%p GMT")
                date_val = dt.strftime("%Y-%m-%d")
                time_val = dt.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                # Fallback for unexpected format
                parts = start_date_str.split(" ", 1)
                date_val = parts[0]
                time_val = parts[1] if len(parts) > 1 else ""

        flat = {
            # document (top-level identifiers)
            "file_name": doc.get("file_name"),
            "company_name": doc.get("company_name"),
            "company_ticker": doc.get("company_ticker"),
            "date": date_val,
            "time": time_val,
            "quarter": doc.get("quarter"),
            "city": doc.get("city"),
            "event_title": doc.get("event_title"),

            # document attributes
            "sentiment": da.get("sentiment"),
            "ML": da.get("ML"),
            "LM": da.get("LM"),
            "HIV4": da.get("HIV4"),
            "num_sentences": da.get("num_sentences"),
            "num_words": da.get("num_words"),
            "flesch_reading_ease": da.get("flesch_reading_ease"),
            "flesch_kincaid_grade": da.get("flesch_kincaid_grade"),
            "smog_index": da.get("smog_index"),
            "coleman_liau_index": da.get("coleman_liau_index"),
            "automated_readability_index": da.get("automated_readability_index"),
            "dale_chall_readability_score": da.get("dale_chall_readability_score"),
            "difficult_words": da.get("difficult_words"),
            "linsear_write_formula": da.get("linsear_write_formula"),
            "gunning_fog": da.get("gunning_fog"),
            "text_standard": da.get("text_standard"),
        }
        return flat

    def process_new_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = "results",
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
        pattern: str = "*.xml",
        recursive: bool = False,
        csv_name: str = "batch_summary.csv",
        skip_on_error: bool = True,
        specific_files: Optional[Sequence[Union[str, Path]]] = None,
        batch_folder_name: Optional[str] = None,
        concurrent_io: bool = False,
        num_threads: int = 1,
    ) -> Dict[str, Any]:
        """
        Process a batch of earnings call transcripts from raw XML files.

        This method orchestrates the analysis of multiple files, supporting parallel processing,
        sentiment analysis, and result aggregation.

        Args:
            input_dir (Union[str, Path]): Directory containing the raw XML transcript files.
            output_dir (Union[str, Path], optional): Base directory where results will be saved. Defaults to "results".
            setup_dict (Optional[Dict[str, Any]], optional): Configuration dictionary for the analysis models (e.g., sentiment model). Defaults to None.
            run_sentiment (bool, optional): Whether to perform sentiment analysis on the transcripts. Defaults to False.
            matching_method (Optional[str], optional): The method to use for keyword matching (e.g., "direct", "cosine"). Defaults to None.
            pattern (str, optional): Glob pattern to match files within input_dir. Defaults to "*.xml".
            recursive (bool, optional): Whether to search input_dir recursively. Defaults to False.
            csv_name (str, optional): Name of the summary CSV file to generate. Defaults to "batch_summary.csv".
            skip_on_error (bool, optional): If True, continues processing other files if one fails. Defaults to True.
            specific_files (Optional[Sequence[Union[str, Path]]], optional): List of specific file paths to process, ignoring input_dir scanning. Defaults to None.
            batch_folder_name (Optional[str], optional): Specific name for the batch output folder. If None, a timestamped folder is created. Defaults to None.
            concurrent_io (bool, optional): If True, enables concurrent I/O operations (writing results) alongside processing. Defaults to False.
            num_threads (int, optional): Number of threads to use for parallel processing. Defaults to 1. Only works if running on a free-threaded python build.

        Returns:
            Dict[str, Any]: A summary dictionary containing paths to the results, counts of processed files, and any errors encountered.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found or not a directory: {input_dir}")

        if run_sentiment:
            # Log model info
            if setup_dict and "hf_model" in setup_dict:
                model_name = setup_dict["hf_model"]
            elif self.analyst.setup:
                model_name = getattr(self.analyst.setup.transformer.model, "name_or_path", "unknown")
            else:
                model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
            print(f"Using HuggingFace model for sentiment: {model_name}")
            
            # Print batch size
            if self.analyst.setup:
                print(f"Batch size: {self.analyst.setup.batch_size}")
            elif setup_dict:
                print(f"Batch size: {setup_dict.get('batch_size', 'auto')}")
            else:
                print(f"Batch size: auto")

        # Create a single batch folder
        if batch_folder_name:
            base_batch_dir = Path(output_dir) / batch_folder_name
            batch_dir = base_batch_dir
            counter = 1
            while batch_dir.exists():
                batch_dir = Path(output_dir) / f"{batch_folder_name}_{counter}"
                counter += 1
        else:
            batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = Path(output_dir) / f"batch_{batch_stamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Gather files
        if specific_files is not None:
            files = []
            for path_like in specific_files:
                candidate = Path(path_like)
                if not candidate.exists() or not candidate.is_file():
                    raise FileNotFoundError(f"Specified file not found: {candidate}")
                files.append(candidate)
        else:
            files = (
                list(input_dir.rglob(pattern)) if recursive
                else list(input_dir.glob(pattern))
            )
            files = [p for p in files if p.is_file()]

        if not files:
            raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir} (recursive={recursive}).")

        rows = []
        results = []
        errors = []
        files_sorted = sorted(files)
        
        effective_threads = self._resolve_num_threads(num_threads)
        use_threading = effective_threads > 1 or concurrent_io

        if use_threading:
            # Queue: (analysis_data, output_dir_str, file_path)
            write_queue = queue.Queue(maxsize=50)
            thread_results = []
            thread_errors = []
            
            mode_desc = f"Multi-threaded ({effective_threads} threads)" if effective_threads > 1 else "Concurrent I/O"
            print(f"{mode_desc} enabled: Matching and Writing will run in parallel.")
            # Use file=sys.stderr for better Windows terminal compatibility
            pbar_processing = tqdm(total=len(files_sorted), desc="Processing (CPU)", unit="file", position=0, dynamic_ncols=True, file=sys.stderr)
            pbar_saving = tqdm(total=len(files_sorted), desc="Saving/Uploading (I/O)", unit="file", position=1, dynamic_ncols=True, file=sys.stderr)
            
            def writer_worker():
                while True:
                    item = write_queue.get()
                    if item is None:
                        write_queue.task_done()
                        break
                    
                    data, out_dir, fpath = item
                    try:
                        # Assumes Analyst has public `save_results`
                        res = self.analyst.save_results(data, out_dir)
                        thread_results.append((res, fpath))
                    except Exception as e:
                        thread_errors.append((e, fpath))
                    finally:
                        write_queue.task_done()
                        pbar_saving.update(1)

            t = threading.Thread(target=writer_worker, daemon=True)
            t.start()

            if effective_threads > 1:
                # Parallel Processing
                # Capture needed state for new Analyst instances
                current_setups = self.analyst.setups
                current_keyword_path = self.analyst.keyword_path

                def process_worker(f_path: Path):
                    # Create thread-local analyst to avoid state corruption
                    # We reuse setups (model) assuming thread-safety of the model/setup, 
                    # but MatchingAgent (recreated) is safe.
                    local_analyst = Analyst(setups=current_setups, keyword_path=current_keyword_path)
                    return local_analyst.analyze_document_memory(
                        earnings_call_path=f_path,
                        setup_dict=setup_dict,
                        run_sentiment=run_sentiment,
                        matching_method=matching_method,
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_threads) as executor:
                    future_to_file = {executor.submit(process_worker, fp): fp for fp in files_sorted}
                    
                    for future in concurrent.futures.as_completed(future_to_file):
                        fpath = future_to_file[future]
                        try:
                            data = future.result()
                            write_queue.put((data, str(batch_dir), fpath))
                        except Exception as e:
                            if skip_on_error:
                                error_message = str(e)
                                rows.append({"file_name": fpath.name, "file_path": str(fpath), "error": error_message})
                                errors.append({"file_name": fpath.name, "file_path": str(fpath), "reason": error_message})
                                pbar_saving.update(1) # We skip the saving step for this file
                            else:
                                # Stop everything
                                # We can't easily cancel other futures, but we can raise
                                raise e
                        finally:
                            pbar_processing.update(1)
            else:
                # Single thread processing (concurrent_io only)
                for fpath in files_sorted:
                    try:
                        data = self.analyst.analyze_document_memory(
                            earnings_call_path=fpath,
                            setup_dict=setup_dict,
                            run_sentiment=run_sentiment,
                            matching_method=matching_method,
                        )
                        write_queue.put((data, str(batch_dir), fpath))
                    except Exception as e:
                        if skip_on_error:
                            error_message = str(e)
                            rows.append({"file_name": fpath.name, "file_path": str(fpath), "error": error_message})
                            errors.append({"file_name": fpath.name, "file_path": str(fpath), "reason": error_message})
                            pbar_saving.update(1) 
                            continue
                        else:
                            pbar_processing.close()
                            pbar_saving.close()
                            raise
                    finally:
                        pbar_processing.update(1)

            write_queue.put(None)
            t.join()
            pbar_processing.close()
            pbar_saving.close()

            # Process collected results
            for res, fpath in thread_results:
                lightweight_res = {k: v for k, v in res.items() if k != "doc_fit"}
                results.append(lightweight_res)
                
                toml_path = Path(res["toml_path"])
                with open(toml_path, "r", encoding="utf-8") as tf:
                    tdata = toml.load(tf)
                row = self._flatten_analysis_toml(tdata)
                rows.append(row)
            
            for exc, fpath in thread_errors:
                if skip_on_error:
                    error_message = str(exc)
                    rows.append({"file_name": fpath.name, "file_path": str(fpath), "error": error_message})
                    errors.append({"file_name": fpath.name, "file_path": str(fpath), "reason": error_message})
                else:
                    raise exc

        else:
            # Sequential (No Threading, No Concurrent IO)
            for fpath in tqdm(files_sorted, total=len(files_sorted), desc="Processing files", unit="file", dynamic_ncols=True, file=sys.stderr):
                try:
                    # Use the public wrapper process_single_document
                    res = self.analyst.process_single_document(
                        earnings_call_path=fpath,
                        setup_dict=setup_dict,
                        run_sentiment=run_sentiment,
                        matching_method=matching_method,
                        output_dir=str(batch_dir),
                    )
                    
                    lightweight_res = {k: v for k, v in res.items() if k != "doc_fit"}
                    results.append(lightweight_res)

                    toml_path = Path(res["toml_path"])
                    with open(toml_path, "r", encoding="utf-8") as tf:
                        tdata = toml.load(tf)

                    row = self._flatten_analysis_toml(tdata)
                    rows.append(row)

                except Exception as e:
                    if skip_on_error:
                        error_message = str(e)
                        rows.append({"file_name": fpath.name, "file_path": str(fpath), "error": error_message})
                        errors.append({"file_name": fpath.name, "file_path": str(fpath), "reason": error_message})
                        continue
                    else:
                        raise

        # Write CSV
        csv_path = batch_dir / csv_name
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in fieldnames})

        return {
            "batch_directory": str(batch_dir),
            "csv_path": str(csv_path),
            "num_files_processed": len(files),
            "results": results,
            "errors": errors,
        }

    def process_batch_to_supabase(
        self,
        input_dir: Union[str, Path],
        year: int,
        supabase_client,
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
        pattern: str = "*.xml",
        recursive: bool = False,
        skip_on_error: bool = True,
        specific_files: Optional[Sequence[Union[str, Path]]] = None,
        num_threads: int = 1,
    ) -> Dict[str, Any]:
        """
        Process earnings calls and upload results directly to Supabase (no intermediate files).
        
        This method bypasses file creation entirely, uploading analysis results directly
        to the Supabase database as each document is processed.
        
        Args:
            input_dir (Union[str, Path]): Directory containing XML transcript files
            year (int): Year of the earnings calls (for database storage)
            supabase_client: Initialized Supabase client
            setup_dict (Optional[Dict[str, Any]]): Setup configuration for sentiment analysis
            run_sentiment (bool): Whether to run sentiment analysis. Defaults to False.
            matching_method (Optional[str]): Matching method ('direct' or 'cosine'). Defaults to None.
            pattern (str): Glob pattern to match files. Defaults to "*.xml".
            recursive (bool): Whether to search recursively. Defaults to False.
            skip_on_error (bool): Continue on errors. Defaults to True.
            specific_files (Optional[Sequence]): Specific files to process. Defaults to None.
            num_threads (int): Number of threads for parallel processing. Defaults to 1.
        
        Returns:
            Dict[str, Any]: Summary with upload statistics and errors
        """
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        if run_sentiment:
            if setup_dict and "hf_model" in setup_dict:
                model_name = setup_dict["hf_model"]
            elif self.analyst.setup:
                model_name = getattr(self.analyst.setup.transformer.model, "name_or_path", "unknown")
            else:
                model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
            print(f"Using HuggingFace model for sentiment: {model_name}")
            
            if self.analyst.setup:
                print(f"Batch size: {self.analyst.setup.batch_size}")
            elif setup_dict:
                print(f"Batch size: {setup_dict.get('batch_size', 'auto')}")
            else:
                print(f"Batch size: auto")

        # Initialize Supabase uploader
        uploader = SupabaseUploader(supabase_client, batch_size=500)
        print("Initializing company cache from Supabase...")
        num_companies = uploader.initialize_company_cache()
        print(f"✓ Loaded {num_companies} existing companies")

        # Gather files
        if specific_files is not None:
            files = []
            for path_like in specific_files:
                candidate = Path(path_like)
                if not candidate.exists() or not candidate.is_file():
                    raise FileNotFoundError(f"Specified file not found: {candidate}")
                files.append(candidate)
        else:
            files = (
                list(input_dir.rglob(pattern)) if recursive
                else list(input_dir.glob(pattern))
            )
            files = [p for p in files if p.is_file()]

        if not files:
            raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir}")

        upload_results = []
        errors = []
        files_sorted = sorted(files)
        
        effective_threads = self._resolve_num_threads(num_threads)
        use_threading = effective_threads > 1

        if use_threading:
            # Multi-threaded processing with direct upload
            print(f"Multi-threaded ({effective_threads} threads) enabled: Processing + Uploading in parallel")
            pbar_processing = tqdm(total=len(files_sorted), desc="Processing & Uploading", unit="file", position=0, dynamic_ncols=True, file=sys.stderr)
            
            # Capture needed state
            current_setups = self.analyst.setups
            current_keyword_path = self.analyst.keyword_path

            def process_and_upload(f_path: Path):
                # Create thread-local analyst
                local_analyst = Analyst(setups=current_setups, keyword_path=current_keyword_path)
                
                # Analyze
                data = local_analyst.analyze_document_memory(
                    earnings_call_path=f_path,
                    setup_dict=setup_dict,
                    run_sentiment=run_sentiment,
                    matching_method=matching_method,
                )
                
                # Upload directly (note: uploader is shared, but upsert is thread-safe)
                result = uploader.upload_document(data, year)
                result["file_path"] = str(f_path)
                return result

            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_threads) as executor:
                future_to_file = {executor.submit(process_and_upload, fp): fp for fp in files_sorted}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    fpath = future_to_file[future]
                    try:
                        result = future.result()
                        upload_results.append(result)
                        if result["status"] != "success":
                            errors.append({"file": str(fpath), "reason": result.get("reason", "Unknown")})
                    except Exception as e:
                        if skip_on_error:
                            errors.append({"file": str(fpath), "reason": str(e)})
                        else:
                            raise e
                    finally:
                        pbar_processing.update(1)
            
            pbar_processing.close()

        else:
            # Sequential processing with direct upload
            for fpath in tqdm(files_sorted, total=len(files_sorted), desc="Processing & Uploading", unit="file", dynamic_ncols=True, file=sys.stderr):
                try:
                    # Analyze
                    data = self.analyst.analyze_document_memory(
                        earnings_call_path=fpath,
                        setup_dict=setup_dict,
                        run_sentiment=run_sentiment,
                        matching_method=matching_method,
                    )
                    
                    # Upload directly
                    result = uploader.upload_document(data, year)
                    result["file_path"] = str(fpath)
                    upload_results.append(result)
                    
                    if result["status"] != "success":
                        errors.append({"file": str(fpath), "reason": result.get("reason", "Unknown")})

                except Exception as e:
                    if skip_on_error:
                        errors.append({"file": str(fpath), "reason": str(e)})
                        continue
                    else:
                        raise

        # Finalize uploader (flush any buffered data)
        uploader.finalize()
        
        # Count successful uploads
        successful = sum(1 for r in upload_results if r["status"] == "success")
        
        return {
            "year": year,
            "num_files_processed": len(files),
            "num_uploaded": successful,
            "num_errors": len(errors),
            "upload_results": upload_results,
            "errors": errors,
        }

    def process_existing_batch(
        self,
        batch_dir: Union[str, Path],
        keyword_path: str,
        matching_method: str,
        skip_on_error: bool = True,
        transcript_roots: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        search_recursive: bool = True,
        concurrent_io: bool = False,
        num_threads: int = 1,
        check_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Run keyword matching analysis on an already processed batch of transcripts.

        This method is useful when you have already parsed the documents and want to run
        different keyword sets or matching algorithms without re-parsing the original files.

        Args:
            batch_dir (Union[str, Path]): Directory containing the previously processed results (should contain subdirectories with analysis_metadata.toml).
            keyword_path (str): Path to the CSV file containing keywords to match.
            matching_method (str): The matching algorithm to use ('direct' or 'cosine').
            skip_on_error (bool, optional): If True, continues processing if a single file/directory fails. Defaults to True.
            transcript_roots (Optional[Union[str, Path, Sequence[Union[str, Path]]]], optional): Directories to search for the original XML transcripts if not found in default locations. Defaults to None.
            search_recursive (bool, optional): Whether to search transcript_roots recursively. Defaults to True.
            concurrent_io (bool, optional): If True, enables concurrent I/O for saving results. Defaults to False.
            num_threads (int, optional): Number of threads for parallel matching. Defaults to 1.
            check_state (bool, optional): If True, checks for existing results and resumes. Defaults to True.

        Returns:
            Dict[str, Any]: A summary dictionary containing the match ID, output CSV path, and results.
        """
        batch_dir = Path(batch_dir)
        if not batch_dir.exists() or not batch_dir.is_dir():
            raise NotADirectoryError(f"Batch directory not found: {batch_dir}")

        if matching_method not in ("direct", "cosine"):
            raise ValueError("matching_method must be 'direct' or 'cosine'")

        # -- Path Resolution Logic --
        resolved_roots: List[Path] = []
        current = batch_dir
        project_root = None
        for _ in range(6):
            if (current / "data").exists():
                project_root = current
                break
            current = current.parent

        if transcript_roots is not None:
            if isinstance(transcript_roots, (str, Path)):
                transcript_roots = [transcript_roots]
            for root in transcript_roots:
                try:
                    resolved_roots.append(Path(root))
                except TypeError:
                    continue
        if project_root is not None:
            resolved_roots.append(project_root / "data" / "earnings_calls")

        # Deduplicate
        seen_roots = set()
        filtered_roots: List[Path] = []
        for root in resolved_roots:
            if root in seen_roots: continue
            seen_roots.add(root)
            filtered_roots.append(root)
        resolved_roots = filtered_roots

        transcript_lookup: Dict[str, Path] = {}
        indexed_roots: set[Path] = set()

        def find_transcript_by_name(file_name: str) -> Optional[Path]:
            nonlocal transcript_lookup
            if not file_name: return None
            for root in resolved_roots:
                if not root.exists(): continue
                if root not in indexed_roots:
                    if search_recursive:
                        try:
                            for candidate in root.rglob("*.xml"):
                                transcript_lookup.setdefault(candidate.name, candidate)
                        except (OSError, ValueError): pass
                    else:
                        for candidate in root.glob("*.xml"):
                            transcript_lookup.setdefault(candidate.name, candidate)
                    indexed_roots.add(root)
                candidate_path = transcript_lookup.get(file_name)
                if candidate_path and candidate_path.exists():
                    return candidate_path
            return None

        # -- Setup Matching --
        from src.analysis.match_extraction.matching_agent import MatchingAgent
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        keyword_name = Path(keyword_path).stem
        match_id = f"{keyword_name}_{timestamp}"

        subdirs = [d for d in batch_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {batch_dir}")

        results = []
        exposure_rows = []
        sorted_subdirs = sorted(subdirs)

        def generate_csv_row(res_dict):
            toml_d = res_dict["toml_data"]
            exp_res = res_dict["exposure_results"]
            doc_metadata = toml_d.get("document", {})
            
            # Simple date parsing for summary
            start_date_str = doc_metadata.get("start_date")
            date_val, time_val = None, None
            if start_date_str:
                parts = start_date_str.split(" ", 1)
                date_val = parts[0]
                time_val = parts[1] if len(parts) > 1 else ""

            return {
                "file_name": doc_metadata.get("file_name"),
                "company_name": doc_metadata.get("company_name"),
                "company_ticker": doc_metadata.get("company_ticker"),
                "date": date_val,
                "time": time_val,
                "quarter": doc_metadata.get("quarter"),
                "match_id": res_dict["match_id"],
                "keyword_name": res_dict["keyword_name"],
                "matching_method": res_dict["matching_method"],
                "total_keywords_searched": exp_res.total_keywords_searched,
                "total_keywords_with_matches": exp_res.total_keywords_with_matches,
                "total_direct_matches": exp_res.total_direct_matches,
                "total_cosine_matches": exp_res.total_cosine_matches,
                "total_matches": exp_res.total_direct_matches + exp_res.total_cosine_matches,
            }

        def _check_existing_match(toml_data: Dict[str, Any], subdir: Path) -> Optional[Dict[str, Any]]:
            """
            Checks if a matching run with the same keyword_name and matching_method exists.
            If found and valid, returns the data package with loaded results.
            """
            from src.analysis.match_extraction.exposure_results import ExposureResults
            matching_runs = toml_data.get("metadata_matching_runs", [])
            for run in matching_runs:
                if (run.get("keyword_name") == keyword_name and 
                    run.get("matching_method") == matching_method):
                    
                    # Found a potential match, check file existence
                    exp_path_str = run.get("exposure_results_path")
                    if not exp_path_str: continue
                    
                    exp_path = Path(exp_path_str)
                    if not exp_path.is_absolute():
                        exp_path = subdir / exp_path
                        
                    if exp_path.exists():
                        try:
                            exposure_results = ExposureResults.load_json(str(exp_path))
                            return {
                                "subdir": subdir,
                                "exposure_results": exposure_results,
                                "toml_data": toml_data,
                                "match_id": run.get("match_id"),
                                "keyword_path": keyword_path,
                                "keyword_name": keyword_name,
                                "matching_method": matching_method,
                                "timestamp": run.get("timestamp"),
                                "already_processed": True
                            }
                        except Exception as e:
                            print(f"Warning: Failed to load existing results at {exp_path}: {e}")
                            pass
            return None

        # -- Pre-scan for existing results --
        pending_subdirs = []
        already_processed_count = 0
        
        if check_state:
            print(f"Scanning {len(sorted_subdirs)} files for existing results...")
            
            # Use a thread pool for faster checking of existing files if there are many
            with concurrent.futures.ThreadPoolExecutor() as scanner:
                # Helper to return (subdir, existing_package)
                def scan_subdir(sd):
                    tp = sd / "analysis_metadata.toml"
                    if not tp.exists():
                        return sd, None
                    try:
                        with open(tp, "r", encoding="utf-8") as f:
                            td = toml.load(f)
                        existing = _check_existing_match(td, sd)
                        return sd, existing
                    except Exception:
                        return sd, None

                # Execute scan
                future_to_subdir = {scanner.submit(scan_subdir, sd): sd for sd in sorted_subdirs}
                
                # We want to maintain order for consistency if possible, but results list order doesn't strictly matter
                # pending_subdirs needs to be processed.
                
                # Temporary storage to sort back later if needed, or just append
                # For progress bar consistency, simple append is fine.
                
                for future in tqdm(concurrent.futures.as_completed(future_to_subdir), total=len(sorted_subdirs), desc="Restoring state", unit="file", dynamic_ncols=True, file=sys.stderr):
                    sd, existing_pkg = future.result()
                    if existing_pkg:
                        results.append(existing_pkg)
                        exposure_rows.append(generate_csv_row(existing_pkg))
                        already_processed_count += 1
                    else:
                        pending_subdirs.append(sd)
            
            print(f"Found {already_processed_count} existing results. Resuming processing for {len(pending_subdirs)} files.")
        
        else:
            pending_subdirs = sorted_subdirs
            print(f"Checking disabled. Processing all {len(pending_subdirs)} files.")
        
        # Sort pending to maintain deterministic processing order
        pending_subdirs.sort()

        # -- Execution --
        effective_threads = self._resolve_num_threads(num_threads)
        use_threading = effective_threads > 1 or concurrent_io

        if use_threading:
            write_queue = queue.Queue(maxsize=50)
            thread_results = []
            thread_errors = []
            
            mode_desc = f"Multi-threaded ({effective_threads} threads)" if effective_threads > 1 else "Concurrent I/O"
            print(f"{mode_desc} enabled: Matching and Writing will run in parallel.")

            # Use file=sys.stderr for better Windows terminal compatibility
            pbar_processing = tqdm(total=len(sorted_subdirs), initial=already_processed_count, desc="Matching (CPU)", unit="file", position=0, dynamic_ncols=True, file=sys.stderr)
            pbar_saving = tqdm(total=len(sorted_subdirs), initial=already_processed_count, desc="Saving/Uploading (I/O)", unit="file", position=1, dynamic_ncols=True, file=sys.stderr)
            
            def writer_worker():
                while True:
                    item = write_queue.get()
                    if item is None:
                        write_queue.task_done()
                        break
                    try:
                        if item.get("already_processed"):
                            thread_results.append(item)
                        else:
                            # Assumes Analyst has public `save_match_results`
                            res = self.analyst.save_match_results(item)
                            thread_results.append(res)
                    except Exception as e:
                        thread_errors.append((e, item.get("subdir")))
                    finally:
                        write_queue.task_done()
                        pbar_saving.update(1)

            t = threading.Thread(target=writer_worker, daemon=True)
            t.start()

            def process_subdir(subdir: Path):
                toml_path = subdir / "analysis_metadata.toml"
                if not toml_path.exists():
                    return None

                with open(toml_path, "r", encoding="utf-8") as f:
                    toml_data = toml.load(f)

                # NOTE: We already checked for existing match in the pre-scan, 
                # so we can skip calling _check_existing_match here.

                doc_meta = toml_data.get("document", {})
                earnings_call_path = doc_meta.get("file_path")
                file_name = doc_meta.get("file_name")

                resolved_path: Optional[Path] = None
                if earnings_call_path and Path(earnings_call_path).exists():
                    resolved_path = Path(earnings_call_path)
                elif file_name:
                    resolved_path = find_transcript_by_name(file_name)

                if resolved_path is None:
                    raise FileNotFoundError(f"Earnings call not found for {subdir.name}")

                # Update path
                earnings_call_path = str(resolved_path)
                doc_meta["file_path"] = earnings_call_path
                toml_data["document"] = doc_meta

                # Create local agent for thread safety if using multiple threads
                local_matching_agent = MatchingAgent(keywords_path=keyword_path)

                # Run Matching
                exposure_results = local_matching_agent.single_processing(
                    document_path=earnings_call_path,
                    matching_function=matching_method
                )

                return {
                    "subdir": subdir,
                    "exposure_results": exposure_results,
                    "toml_data": toml_data,
                    "match_id": match_id,
                    "keyword_path": keyword_path,
                    "keyword_name": keyword_name,
                    "matching_method": matching_method,
                    "timestamp": timestamp,
                }

            if effective_threads > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_threads) as executor:
                    future_to_subdir = {executor.submit(process_subdir, sd): sd for sd in pending_subdirs}
                    for future in concurrent.futures.as_completed(future_to_subdir):
                        sd = future_to_subdir[future]
                        try:
                            data_package = future.result()
                            if data_package:
                                write_queue.put(data_package)
                            else:
                                pbar_saving.update(1)
                        except Exception as e:
                            if skip_on_error:
                                thread_errors.append((e, sd))
                                pbar_saving.update(1)
                            else:
                                raise e
                        finally:
                            pbar_processing.update(1)
            else:
                # Single thread processing (concurrent_io only)
                for subdir in pending_subdirs:
                    try:
                        data_package = process_subdir(subdir)
                        if data_package:
                            write_queue.put(data_package)
                        else:
                            pbar_saving.update(1)
                    except Exception as e:
                        if skip_on_error:
                            thread_errors.append((e, subdir))
                            pbar_saving.update(1)
                        else:
                            raise e
                    finally:
                        pbar_processing.update(1)

            write_queue.put(None)
            t.join()
            pbar_processing.close()
            pbar_saving.close()

            for res in thread_results:
                results.append(res)
                exposure_rows.append(generate_csv_row(res))
            for exc, subdir_path in thread_errors:
                if skip_on_error:
                    exposure_rows.append({"file_name": subdir_path.name if subdir_path else "unknown", "error": str(exc)})
                else:
                    raise exc

        else:
            # Sequential (No Threading, No Concurrent IO)
            matching_agent = MatchingAgent(keywords_path=keyword_path)
            
            # Initialize tqdm with total subdirs but start at already_processed_count
            # Iterate only over pending_subdirs
            for subdir in tqdm(pending_subdirs, total=len(sorted_subdirs), initial=already_processed_count, desc="Matching keywords", unit="file", dynamic_ncols=True, file=sys.stderr):
                toml_path = subdir / "analysis_metadata.toml"
                if not toml_path.exists():
                    if skip_on_error: continue
                    else: raise FileNotFoundError(f"analysis_metadata.toml not found in {subdir}")

                try:
                    with open(toml_path, "r", encoding="utf-8") as f:
                        toml_data = toml.load(f)

                    # NOTE: Pre-scan already handled existing matches
                    
                    doc_meta = toml_data.get("document", {})
                    earnings_call_path = doc_meta.get("file_path")
                    file_name = doc_meta.get("file_name")

                    resolved_path: Optional[Path] = None
                    if earnings_call_path and Path(earnings_call_path).exists():
                        resolved_path = Path(earnings_call_path)
                    elif file_name:
                        resolved_path = find_transcript_by_name(file_name)

                    if resolved_path is None:
                        if skip_on_error:
                            exposure_rows.append({"file_name": file_name or subdir.name, "error": "Earnings call not found"})
                            continue
                        else:
                            raise FileNotFoundError(f"Earnings call not found")

                    earnings_call_path = str(resolved_path)
                    doc_meta["file_path"] = earnings_call_path
                    toml_data["document"] = doc_meta

                    exposure_results = matching_agent.single_processing(
                        document_path=earnings_call_path,
                        matching_function=matching_method
                    )
                    
                    data_package = {
                        "subdir": subdir,
                        "exposure_results": exposure_results,
                        "toml_data": toml_data,
                        "match_id": match_id,
                        "keyword_path": keyword_path,
                        "keyword_name": keyword_name,
                        "matching_method": matching_method,
                        "timestamp": timestamp,
                    }
                    res = self.analyst.save_match_results(data_package)
                    results.append(res)
                    exposure_rows.append(generate_csv_row(res))

                except Exception as e:
                    if skip_on_error:
                        exposure_rows.append({"file_name": subdir.name, "error": str(e)})
                        continue
                    else:
                        raise

        # Write CSV
        csv_filename = f"exposure_summary_{match_id}.csv"
        csv_path = batch_dir / csv_filename
        all_keys = set()
        for r in exposure_rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in exposure_rows:
                writer.writerow({k: r.get(k) for k in fieldnames})

        return {
            "batch_directory": str(batch_dir),
            "match_id": match_id,
            "exposure_summary_csv": str(csv_path),
            "num_files_processed": len(results),
            "results": results,
        }

    def check_integrity(
        self,
        earnings_calls_dir: Union[str, Path],
        results_dir: Union[str, Path],
        *,
        metadata_filename: str = "analysis_metadata.toml",
        save_results: bool = False,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Verify the completeness of a processing batch by comparing results against source files.

        Checks if every source transcript in `earnings_calls_dir` has a corresponding
        result in `results_dir`.

        Args:
            earnings_calls_dir (Union[str, Path]): Directory containing the source XML transcript files.
            results_dir (Union[str, Path]): Directory containing the processed results.
            metadata_filename (str, optional): Name of the metadata file to look for in result directories. Defaults to "analysis_metadata.toml".
            save_results (bool, optional): Whether to save the integrity report to a JSON file. Defaults to False.
            save_path (Optional[Union[str, Path]], optional): Specific path to save the integrity report. If None, saves to 'integrity_snapshot.json' in results_dir. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary detailing present/missing files, total counts, and specific missing file details.
        """
        earnings_dir = Path(earnings_calls_dir).expanduser()
        results_dir = Path(results_dir).expanduser()

        if not earnings_dir.exists() or not earnings_dir.is_dir():
            raise NotADirectoryError(f"Earnings calls directory not found: {earnings_dir}")
        if not results_dir.exists() or not results_dir.is_dir():
            raise NotADirectoryError(f"Results directory not found: {results_dir}")

        transcript_lookup: Dict[str, Path] = {}
        for xml_file in earnings_dir.rglob("*.xml"):
            if not xml_file.is_file(): continue
            transcript_lookup.setdefault(xml_file.name, xml_file)

        if not transcript_lookup:
            raise FileNotFoundError(f"No XML files found in {earnings_dir}")

        metadata_paths = [p for p in results_dir.rglob(metadata_filename) if p.is_file()]
        if not metadata_paths:
            raise FileNotFoundError(f"No '{metadata_filename}' files found in {results_dir}")

        analyzed_files: set[str] = set()
        metadata_issues = []

        for metadata_path in tqdm(metadata_paths, total=len(metadata_paths), desc="Checking integrity", unit="file", dynamic_ncols=True, file=sys.stderr):
            try:
                toml_data = toml.load(metadata_path)
            except (OSError, toml.TomlDecodeError) as exc:
                metadata_issues.append({"file_name": None, "metadata_path": str(metadata_path), "reason": f"TOML error: {exc}"})
                continue

            file_name = toml_data.get("document", {}).get("file_name")
            if not file_name:
                metadata_issues.append({"file_name": None, "metadata_path": str(metadata_path), "reason": "Missing document.file_name"})
                continue

            if file_name not in transcript_lookup:
                metadata_issues.append({"file_name": file_name, "metadata_path": str(metadata_path), "reason": "File not found in earnings_calls_dir"})
                continue

            analyzed_files.add(file_name)

        missing_details = [
            {"file_name": name, "file_path": str(path)}
            for name, path in sorted(transcript_lookup.items())
            if name not in analyzed_files
        ]

        result: Dict[str, Any] = {
            "present": len(analyzed_files),
            "missing": len(missing_details),
            "total_transcripts": len(transcript_lookup),
            "total_metadata_files": len(metadata_paths),
            "missing_details": missing_details,
            "metadata_issues": metadata_issues,
            "earnings_calls_dir": str(earnings_dir),
            "results_dir": str(results_dir),
            "metadata_sample_path": str(metadata_paths[0]) if metadata_paths else None,
        }

        if save_results:
            out_path = Path(save_path).expanduser() if save_path else results_dir / "integrity_snapshot.json"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                result["saved_to"] = str(out_path)
            except Exception as exc:
                result["save_error"] = str(exc)

        return result

    def repair_batch(
        self,
        earnings_calls_dir: Union[str, Path],
        results_dir: Union[str, Path],
        *,
        integrity_snapshot: Optional[Dict[str, Any]] = None,
        integrity_snapshot_path: Optional[Union[str, Path]] = None,
        metadata_filename: str = "analysis_metadata.toml",
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
        csv_name: str = "batch_summary.csv",
        skip_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Attempt to process missing files identified by an integrity check.

        This method finds transcripts that are present in the source directory but missing
        from the results, and processes them to complete the batch.

        Args:
            earnings_calls_dir (Union[str, Path]): Source directory for earnings call transcripts.
            results_dir (Union[str, Path]): Directory containing partial results.
            integrity_snapshot (Optional[Dict[str, Any]], optional): Pre-computed integrity report from `check_integrity`. Defaults to None.
            integrity_snapshot_path (Optional[Union[str, Path]], optional): Path to a saved integrity report JSON file. Defaults to None.
            metadata_filename (str, optional): Name of the metadata file to look for. Defaults to "analysis_metadata.toml".
            setup_dict (Optional[Dict[str, Any]], optional): Configuration for analysis (should match original batch). Defaults to None.
            run_sentiment (bool, optional): Whether to run sentiment analysis. Defaults to False.
            matching_method (Optional[str], optional): Matching method to use. Defaults to None.
            csv_name (str, optional): Name of the summary CSV. Defaults to "batch_summary.csv".
            skip_on_error (bool, optional): If True, ignores errors during repair. Defaults to True.

        Returns:
            Dict[str, Any]: A summary of the repair operation, including processed count and updated integrity status.
        """
        earnings_dir = Path(earnings_calls_dir).expanduser()
        results_dir = Path(results_dir).expanduser()

        if integrity_snapshot is not None:
            integrity_before = integrity_snapshot
        elif integrity_snapshot_path is not None:
            snapshot_path = Path(integrity_snapshot_path).expanduser()
            with snapshot_path.open("r", encoding="utf-8") as f:
                integrity_before = json.load(f)
        else:
            integrity_before = self.check_integrity(
                earnings_calls_dir=earnings_dir,
                results_dir=results_dir,
                metadata_filename=metadata_filename,
            )

        missing_calls = integrity_before.get("missing_details") or []
        
        # Helper to find where to put new files (same structure as original batch likely)
        def _infer_analysis_root(base_dir: Path, sample_metadata: Optional[str]) -> Path:
            if not sample_metadata: return base_dir
            metadata_path = Path(sample_metadata)
            try:
                relative = metadata_path.relative_to(base_dir)
            except ValueError:
                return metadata_path.parent
            parts = relative.parts
            if len(parts) >= 3: return base_dir / parts[0]
            if len(parts) >= 2: return base_dir
            return metadata_path.parent

        analysis_root = _infer_analysis_root(results_dir, integrity_before.get("metadata_sample_path"))
        analysis_root.mkdir(parents=True, exist_ok=True)

        valid_missing_records = []
        errors = []
        for detail in missing_calls:
            file_path_str = detail.get("file_path")
            if not file_path_str: continue
            transcript_path = Path(file_path_str)
            if not transcript_path.exists():
                errors.append({"file_name": detail.get("file_name"), "reason": "Transcript not found"})
                continue
            valid_missing_records.append(str(transcript_path))

        processed_count = 0
        if valid_missing_records:
            batch_result = self.process_new_batch(
                input_dir=earnings_dir, # won't be scanned because specific_files is set
                output_dir=str(analysis_root.parent), # process_new_batch creates a subfolder, we want to try and match? 
                # Actually, process_new_batch ALWAYS creates a NEW batch folder. 
                # If we want to add to existing, we probably just want to output to the SAME directory?
                # But process_new_batch logic forces a new timestamped folder unless batch_folder_name is forced.
                # To seamlessly "repair" into the same folder structure is hard if we want strict batch isolation.
                # However, usually repairs are appended as new batches or inside the same big folder.
                # Let's set batch_folder_name to the analysis_root name if possible.
                batch_folder_name=analysis_root.name,
                setup_dict=setup_dict,
                run_sentiment=run_sentiment,
                matching_method=matching_method,
                csv_name=csv_name,
                skip_on_error=skip_on_error,
                specific_files=valid_missing_records,
            )
            processed_count = batch_result["num_files_processed"]
            errors.extend(batch_result.get("errors") or [])
        
        integrity_after = self.check_integrity(
            earnings_calls_dir=earnings_dir,
            results_dir=results_dir,
            metadata_filename=metadata_filename,
        )

        return {
            "processed": processed_count,
            "errors": errors,
            "integrity_before": integrity_before,
            "integrity_after": integrity_after,
        }
