When executing this command: "conda env create -f environment.yml", the error "conda : commande introuvable" was found. To fix it, conda needs to be installed.
1. Download the installer from this website: https://www.anaconda.com/download/
2. Execute "chmod +x /path/to/the/script.sh" to give execution rights to the script
3. Execute "bash SCRIPT_NAME.sh -b" -b: silent installation with default values
4. Execute "source $HOME/anaconda3/bin/activate" to activate conda

When executing "python3 sits.dl.preprocess.py", the error 
"Initial EE initialization failed, attempting authentication: Caller does not have required permission to use project ee-antoinesaget. Grant the caller the roles/serviceusage.serviceUsageConsumer role, or a custom role with the serviceusage.services.use permission, by visiting https://console.developers.google.com/iam-admin/iam/project?project=ee-antoinesaget and then retry. Propagation of the new permission may take a few minutes." is found.
to fix it, we need to create a new project on google earth engine, and change the current name of the project in  the "initialize_earth_engine" function to the new project's name

At the end of the execution:
2025-04-14 17:09:58,151 - INFO - Starting validation
Traceback (most recent call last):
  File "/home/magzoumov/Documents/sits_dl_preprocess/sits_dl_preprocess.py", line 677, in <module>
    main()
  File "/home/magzoumov/Documents/sits_dl_preprocess/sits_dl_preprocess.py", line 668, in main
    df = filter_and_save_valid_parcels(df, processed_arrays_folder, current_dir, logger)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/magzoumov/Documents/sits_dl_preprocess/sits_dl_preprocess.py", line 563, in filter_and_save_valid_parcels
    if (input_folder / f"{idx // 5000}/{idx}.npy").exists():
                          ~~~~^^~~~~~
TypeError: unsupported operand type(s) for //: 'str' and 'int'                                                                                                                                                                                                                                                                             

il faut ajouter un gitignore