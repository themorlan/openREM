from typing import Dict


def default_import(
    func,
    import_type: str,
    file_type_name: str,
    max_other_prcocesses: int,
    max_other_processes_of_type: Dict[str, int],
):
    """
    Executes a default import of a file with an extractor function, the way most extractor scripts do it
    :param func: The extractor function to use
    :param import_type: The string which should be set for task type of the import task
    :file_type_name: A string shown to the user describing which kind of file we need
    :max_other_processes: The maximum number of processes which should be allowed
        concurrently (0 means don't care)
    :max_other_processes_of_type: The maximum number of a certain task type which
         should be allowed to run concurrently with this
    """
    import sys
    import os
    from glob import glob
    from openrem.remapp.tools.background import (
        run_in_background_with_limits,
        wait_task,
    )

    if len(sys.argv) < 2:
        sys.exit(f"Error: Supply at least one argument - {file_type_name}")

    tasks = []
    for arg in sys.argv[1:]:
        for filename in glob(arg):
            filename = os.path.abspath(filename)
            b = run_in_background_with_limits(
                func,
                import_type,
                max_other_prcocesses,
                max_other_processes_of_type,
                filename,
                **{"priority": 10},
            )
            tasks.append(b)

    for t in tasks:
        wait_task(t)
