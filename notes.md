### Etape 1: Refonte de la structure du code

J'ai séparé les fonctions en 4 modules:

- main: main, setup_logging
- data_processing: parse, get_time_windows, process_dataframe,
download_and_process_worker, worker_wrapper, process_parcels,
filter_by_area
- file_operations: filter_and_save_valid_parcels, create_memmap
- earth_engine: initialize_earth_engine, shapely2ee, query, retrieve_data

Puis, pour chaque module (main exclu), j'ai crée une classe:

- data_processing: DataProcessor
- file_operations: FileManager
- earth_engine: EarthEngineClient

Il y a eu quelques erreurs avec la création des modules, puis des classes:

- dans download_and_process_worker, les méthodes shapely2ee, retrieve_data sont appelées, mais ne sont plus dans le même module. J'ai passé la classe ee_client  en argument à process_parcels, puis worker_wrapper et finalement download_and_process_worker.
-> un autre problème était lié à celui-là: parse (dans DataProcessor) est appelé dans shapely2ee, j'ai donc passé le processor en argument à cette méthode (grâce au self)
- la méthode process_dataframe dans DataProcessor a besoin de RADIOMETRIC_BANDS du main, j'ai donc passé cette variable en argument à process_parcels, puis worker_wrapper, download_and_process_worker, et finalement à process_data_frame
- les méthodes parse dans DataProcessor, et query dans EarthEngineClient ont besoin de ALL_BANDS du main, je l'ai également passé en argument à retrieve_data (qui appelle les deux autres méthodes), à download_and_process_worker qui appelle retrieve_data, à worker_wrapper, qui appelle download_and_process_worker, et à process_parcels, qui appelle worker_wrapper

### Etape 2: Gestion de la configuration
J'ai changé l'extension du fichier environnement de yml à yaml: il n'y a pas de différence fonctionnelle, mais c'est l'extension recommendée officiellement ([faq yaml](https://yaml.org/faq.html))

Ensuite, j'ai séparé le code des données de configuration, que j'ai ajouté dans config.yaml. Pour récuperer ces données ensuite, on import yaml (et j'ai ajouté pyyaml dans le environnement.yaml), puis on lit le contenu du fichier et le met dans config avec
```
config = yaml.safe_load(config_file)
```
Par la même occasion, j'ai ajouté les bands dans config.yaml, ce qui facilite leur modification éventuelle.

Pour la définiton des dossiers et chemins jusqu'aux fichiers utilisés, qui étaitent initialement contenue dans le main, j'ai utilisé la librairie argparse, qui me permet de recevoir des arguments au moment de l'execution, et de définir des valeurs par défaut. Par ailleurs, j'ai stocké les valeurs par défaut dans le config.yaml: on peut donc soit modifier ce fichier directement pour changer les dossiers et fichiers utilisés, ou les renter en paramètres au moment de l'execution du main.