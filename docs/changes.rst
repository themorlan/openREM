=======================
OpenREM version history
=======================


1.0.0b1 (2022-xx-xx)
--------------------
* :issue:`937`  Interface: correcting bottom row of exports table
* :issue:`936`  Tasks: added make_skin_map to background tasks on RF RDSR import
* :issue:`934`  DICOM Networking: QR queries are now logged and can be analysed through the web interface
* :issue:`931`  Exports: export RF DAP as float instead of text
* :issue:`928`  Documentation: added restriction in postgres version for earlier OpenREM releases
* :issue:`925`  Docker: docs and config file for enabling bind mounts with SELinux
* :issue:`922`  Database: optimise indexes and duplicate queries
* :issue:`919`  Interface: fixed bug preventing home page listing if study had no date
* :issue:`917`  Interface: added horizontal lines between chart option groups and shaded chart option CheckboxSelectMultiple items
* :issue:`913`  SkinDose: made 2d skin dose map overlay visible by default
* :issue:`911`  Charts: fixed issue with chart data sorting and added label wrap option
* :issue:`910`  SkinDose: fixed rendering of 2d skin dose map with head
* :issue:`909`  Code quality: all model imports absolute
* :issue:`908`  Imports: enabled Device Observer UID to be ignored for specific equipment models when creating display name data during RDSR import
* :issue:`906`  Charts: upgraded Plotly library to latest version
* :issue:`905`  Imports: fixed filter extraction code not working for Siemens Multix DX
* :issue:`904`  Testing: bumped Python image from 3.6 to 3.8
* :issue:`903`  Interface: added patient weight filters to the CT, RF and DX summary pages
* :issue:`901`  Charts: fixed issue where mammography mAs values were displayed 1000x too high on scatter plot
* :issue:`897`  Docker: fixed permissions for PixelMed - now using root throughout
* :issue:`896`  Imports: enabling import of DX with text string in PatientSize field
* :issue:`893`  Charts: fixed issue with over-time charts with many sub-plots failing to plot correctly
* :issue:`888`  SkinDose: added option to support skin dose calculations for listed systems only
* :issue:`886`  Code quality: addressed some SonarCloud issues
* :issue:`882`  SkinDose: added percentage of exposures that interact with phantom
* :issue:`881`  Charts: add option to remove multiple and trailing whitespace in category names
* :issue:`880`  Orthanc: added RF and DX to allowed modalities to enable physics QA images to be kept
* :issue:`879`  Charts: fixed sorting of fluoroscopy charts when split by physician
* :issue:`877`  Charts: added acquisition type restrictions to acquisition-level CT charts
* :issue:`872`  Charts: added ability to split fluoroscopy over-time and histogram charts by physician
* :issue:`871`  Charts: corrected RF chart x-axis labels
* :issue:`870`  Charts: reduced memory footprint of Pandas DataFrame creation
* :issue:`869`  Charts: added doc strings to new chart code
* :issue:`868`  Docker: fixed Python version at 3.8
* :issue:`867`  Documentation: updated chart documentation
* :issue:`866`  Simplified code for different character sets, enabled MultiValue SpecificCharacterSet
* :issue:`865`  Imports: enabled workaround to import Spectrum Dynamics RDSR
* :issue:`864`  Tasks: updated Celery settings for Celery 6.
* :issue:`863`  Interface: removed height and weight from CT study delete
* :issue:`862`  Interface: allow mapping of request, study and acquisition names to standard versions
* :issue:`861`  Interface: added ability to filter mammography on view code, compressed breast thickness and exposure control mode
* :issue:`860`  DICOM Networking: removed built-in DICOM Store SCP functionality
* :issue:`858`  DICOM Networking: query-retrieve logging, filtering and error handling improved
* :issue:`856`  Interface: removed CT acquisition type restriction tick boxes
* :issue:`854`  Interface: added date constraints to links on homepage
* :issue:`853`  Testing: reduced Bitbucket pipeline minutes usage
* :issue:`852`  Code quality: skin dose code formatted with Black
* :issue:`850`  Emails: added oldest study accession number to high fluoro dose alert email subject
* :issue:`849`  Docker: make docker URL an env variable
* :issue:`847`  Documentation: added copy button to commands, added prompts where appropriate
* :issue:`845`  Docker: moved Nginx config to bind folder
* :issue:`844`  Code quality: getting the pipelines right
* :issue:`843`  Code quality: closing target _blank phishing vulnerability
* :issue:`842`  Imports: ContextID code_meaning in make_skin_map and dxdetail
* :issue:`841`  Code quality: format code with Black
* :issue:`840`  Exports: added performing physician to fluoroscopy standard exports
* :issue:`839`  Documentation: removed redundant troubleshooting docs
* :issue:`838`  Imports: fixed issues with changed PersonName behaviour in pydicom 2.0
* :issue:`836`  Installation: updated requirements, Docker and docs for pynetdicom 1.5, pydicom 2.0
* :issue:`835`  Docker: fixed timeout issue with slow pagination
* :issue:`830`  Charts: fixed incorrect histogram data in charts
* :issue:`829`  Installation: added docs for Docker install on computer without internet access
* :issue:`828`  Docker: enabled email configuration to work
* :issue:`827`  SkinDose: made SkinDose results available in OpenREM and made alert triggering possible
* :issue:`826`  Code quality: split views to make more manageable and testable
* :issue:`824`  DICOM Networking: enabled declaration and testing of Orthanc Store SCP in Docker
* :issue:`821`  Code quality: fixed literal comparisons Docker was complaining about
* :issue:`820`  Documentation: converted changes to use sphinx-issues
* :issue:`819`  Removed colons from commands in documentation as they don't format correctly in PDF
* :issue:`818`  Interface: refactored event number filtering
* :issue:`817`  SkinDose: fixed PEP8 and Codacy issues for skinDose
* :issue:`816`  Interface: fixed password change error
* :issue:`815`  Interface: fixed patient name filtering
* :issue:`814`  Deployment: automated deployment to dev.openrem.org and testing.openrem.org reintroduced
* :issue:`808`  Imports: caters for illegal use of mGy units in RDSR for dose at RP values
* :issue:`807`  Exports: fixed errors in PHE fluoro export when values are None
* :issue:`805`  DICOM Networking: fix errors on "association release" and "move complete"
* :issue:`803`  Fixed problem with multiple ModalitiesInStudy entries in c-find response
* :issue:`800`  Tasks: import and export tasks and DICOM queries and moves now listed with more information
* :issue:`799`  DICOM Networking: workaround for stationnames > 16 characters
* :issue:`798`  Exports: prevented error in export view if task_id is missing
* :issue:`797`  Exports: fixed string/byte issues with csv exports
* :issue:`796`  Exports: replaced file() with open() for Python 3.x compatibility
* :issue:`795`  Exports: included acquisition name in PHE radiographic projection export
* :issue:`793`  Installation: added Docker installation
* :issue:`791`  Exports: prevented error when trying to export DX data that has no filter information
* :issue:`790`  Python 3: remove basestring type
* :issue:`789`  Python 3: Median function aggregation code simplified; works with Python 3.7
* :issue:`788`  Tasks: Celery and RabbitMQ dropped, background task processing now managed within Python/OpenREM
* :issue:`787`  Interface: fixed login error
* :issue:`777`  Updated OpenREM to use pydicom 1.3
* :issue:`772`  DICOM Networking: check for station name at series level or study, not both
* :issue:`744`  Added overwrite mode to size import
* :issue:`678`  Enabled import of PX modality panoramic exam data - they appear in the Radiographic section
* :issue:`530`  Updated OpenREM to use pynetdicom 1.4
* :issue:`513`  Internationalization: first translation strings added to documentation
* :issue:`512`  Internationalization: first translation strings added to interface
* :issue:`457`  Updated OpenREM to use Django 2.2
* :issue:`477`  Charts: replaced HighCharts with open source Plotly library
* :issue:`437`  Updated OpenREM to use django-filters v2
* :issue:`433`  Import: Siemens Arcadis Varic dose reports are now imported
* :issue:`404`  Ported OpenREM to Python 3
* :issue:`233`  Charts: added charts of average CTDI and DLP over time
* :issue:`94`   Nuclear medicine: added nuclear medicine SPECT and PET functionality including RRSDR imports

0.10.0 (2019-11-08)
-------------------
* :issue:`785`  Interface: added study level comments to rfdetail.html
* :issue:`784`  Imports: added laterality under target region as per 2017 CP 1676 change
* :issue:`783`  Interface: replaced static links by dynamic versions in rfdetail.html
* :issue:`782`  Exports: fixed RF export issue with studies containing multiple modalities
* :issue:`781`  Charts: fixed issue where charts were mis-labelled if "Case-insensitive categories" was unchecked
* :issue:`780`  Interface: changed mammography accumulated laterality to use code_value rather than code_meaning
* :issue:`779`  Installation: added restriction to django-qsstats-magic version
* :issue:`778`  Imports: added summary field population tests, fixed CT RDSR Total DLP import error
* :issue:`776`  Documentation: grammar and spelling correction for PHE exports
* :issue:`775`  Exports, documentation: fixed units issue and minor docs issue for PHE DX export
* :issue:`774`  Charts: fixed issue where charts were mis-labelled if "Plot a series per system" was unchecked
* :issue:`771`  Interface: entire fluoro exam row now highlighted when dose alert exceeded
* :issue:`770`  Imports: fix to allow non-conformant Opera Swing to import
* :issue:`769`  Interface: modified to allow detail view display of Ziehm studies with missing summary data
* :issue:`768`  Charts: study- and request-level charts now use study-level summary fields to improve performance
* :issue:`765`  Imports: updated error catching to allow Philips BigBore 4DCT RDSR to import
* :issue:`763`  Imports: corrected delta week fluoro study counting for dual plane modalities
* :issue:`762`  Interface: fixed error when deleting dual plane radiography studies
* :issue:`761`  Imports: fixed issue in high dose alert e-mail code where week_delta may be used before assignment
* :issue:`759`  Database: added study level summary fields and migration function
* :issue:`758`  Configuration: corrected issues with location of js_reverse static files
* :issue:`750`  Exports: added export tailored to the 2019 PHE DX dose survey
* :issue:`746`  Imports: enabled import of GE Elite Mini View C-arm RDSR with no template declaration
* :issue:`181`  Imports: corrected import of grid information from RDSRs

0.9.1 (2019-05-16)
------------------
* :issue:`766`  Documentation: updated the Windows Celery documentation to reflect changes required to shutdown Celery 3.1.25
* :issue:`755`  Interface: fix more static URLs to allow virtual directory web server configurations
* :issue:`754`  Documentation and install: updated docs and minimum version for collectstatic_js_reverse
* :issue:`753`  Query-retrieve: removed patient age fields from study level C-FIND that were not used
* :issue:`752`  Exports: fixed missing weight field in PHE CT 2019 export
* :issue:`749`  Documentation: updated the Linux quick install docs
* :issue:`748`  Charts: fixed error that caused blank charts if series per system was selected
* :issue:`747`  Installation: changed minimum Python version for current version of Flower
* :issue:`743`  Testing: added configuration to enable testing with default logging
* :issue:`742`  Interface: sorting of task tables now works in Internet Explorer 11
* :issue:`740`  Installation: fixed Celery version to avoid dependency on Django 1.11
* :issue:`739`  Imports: fixed import errors for GE surgical fluoroscopy
* :issue:`738`  Logging: added single_date query date to log, added tasks aborts to logs
* :issue:`737`  Interface and exports: specify number of events and export to PHE 2019 CT survey specification
* :issue:`736`  Query-retrieve: duplicate study level responses now removed from query
* :issue:`735`  Imports: switched to more secure defusedxml for parsing XML in comments
* :issue:`734`  Query-retrieve: handle illegal image level response with no instance number
* :issue:`732`  Query-retrieve: added advanced option to workaround empty series issue
* :issue:`710`  Interface: time-based columns in Celery and RabbitMQ tables now sorted correctly
* :issue:`404`  Code quality: changes to lead toward Python 3 compliance

0.9.0 (2019-03-06)
------------------
* :issue:`733`  Documentation: post-release fixes for 0.9.0 docs
* :issue:`731`  Imports: fixed another issue with display names on upgrade to 0.9
* :issue:`729`  Interface: replaced hard coded URLs in displaynameview.html and review_failed_imports.html with url names
* :issue:`727`  Imports: fixed issue with display names on upgrade to 0.9
* :issue:`726`  Documentation: updated to include the new task management function
* :issue:`725`  Charts: added fluoroscopy charts of DAP and frequency per requested procedure
* :issue:`723`  Task management: fixed issue with latest version of kombu and amqp on Windows
* :issue:`722`  Interface: dual-plane DX studies are now displayed without error in filtered list and study detail page
* :issue:`721`  Documentation: removed Django Debug Toolbar from default install and documented how to install and use it
* :issue:`720`  Interface: fixed small overlap between skin dose map and irradiation type table
* :issue:`719`  Interface: fixed hardcoded link in template rffiltered.html
* :issue:`717`  Query-retrieve: fixed problem where an error was thrown if association is None
* :issue:`716`  Task manager: removed assumption of queue name from RabbitMQ management interface
* :issue:`714`  Documentation: add missing documentation about changing STATIC_URL if serving OpenREM in a virtual directory
* :issue:`711`  Query-retrieve: fixed problem for zero image series when using -toshiba flag
* :issue:`710`  Interface: Celery and RabbitMQ tables can now be sorted by clicking on column headings
* :issue:`709`  Query-retrieve: corrected query logic for multiple modalities using :issue:`627` Modality tag at study level fix
* :issue:`708`  Query-retrieve: fixed problem for empty Series Number
* :issue:`707`  Interface: fixed issue where sigdig returned an error if it was passed an empty string
* :issue:`706`  Exports: fixed problem where filters were not respected for radiographic exports
* :issue:`705`  Task manager: added Flower to install and integrated to interface
* :issue:`704`  Imports: caters for illegal use of dGy.cm2 units in RDSR for DAP values
* :issue:`703`  Interface: fixed URL lookup error for failed imports on homepage
* :issue:`702`  Query-retrieve: fixed URLs in DICOM javascript files to allow virtual-directories
* :issue:`701`  Interface: made the fluoroscopy exposure detail table sortable by clicking on headers
* :issue:`698`  Imports: allow for incorrect case in Procedure reported tag in RDSR
* :issue:`697`  Testing: added tests for fluoroscopy high dose alerts (single-plane systems)
* :issue:`696`  Interface: fixed broken Delete Studies and Entry button
* :issue:`695`  Imports: added missing name attribute for size_abort url
* :issue:`694`  Query-retrieve: added extensive logging and summary to interface
* :issue:`693`  Interface: fixed display of numbers with significant places settings and comma localisation
* :issue:`691`  Interface: fixed URL lookup error for Display Names page
* :issue:`690`  Interface: added workload stats user option entry back into config menu
* :issue:`689`  Interface: fixed URL lookup error for DICOM summary page
* :issue:`688`  Interface: Add possibility to apply known display name based on Device Observer UID (default: disabled)
* :issue:`685`  Charts: fixed link code that would otherwise cause DLP per acquisition protocol chart histogram links to fail
* :issue:`683`  Installation: added VIRTUAL_DIRECTORY to the settings file to avoid updating local_settings file on upgrade
* :issue:`682`  Charts: fixed problem where links from histogram bars didn't filter correctly when case-insensitive categories selected
* :issue:`681`  Imports: modified RDSR import to work with Varian RDSRs
* :issue:`679`  Interface: added ability to filter CT studies on acquisition type
* :issue:`677`  Interface: added additional filter materials to convert to abbreviations
* :issue:`676`  Imports: improved error handling on patient size imports
* :issue:`675`  Exports: improved resilience when export includes malformed studies
* :issue:`674`  Documentation: amended zip command in example Orthanc configuration to work with Linux and Windows
* :issue:`673`  Imports: handle empty NumericValues and workaround for incorrect Philips Azurion AcquisitionDeviceType
* :issue:`672`  Documentation: improve and extend linux one-page install
* :issue:`670`  Imports: handle illegal multi-value number in Toshiba RDSR with vHP
* :issue:`668`  Code quality: library import and blank space cleanup
* :issue:`667`  Web server: enable OpenREM to be hosted from a non-root folder/virtual-directory
* :issue:`666`  Query-retrieve: handle non-return of ModalitiesInStudy correctly
* :issue:`665`  Interface: added fluoroscopy high dose highlighting and e-mail alerts
* :issue:`662`  Administration: added facility to list and purge RabbitMQ queues
* :issue:`659`  Interface: made the latest study field in summary tables on the home page sort correctly
* :issue:`658`  Interface: added display of workload stats in home page modality tables
* :issue:`637`  Administration: added facility to list and purge RabbitMQ queues
* :issue:`554`  Query-retrieve: added time as matching argument for command line use
* :issue:`461`  Web server: enable OpenREM to be hosted from a non-root folder/virtual-directory (via :issue:`667`)
* :issue:`479`  Administration: added facility to list and delete failed import studies
* :issue:`349`  Task management: fixed issue with Windows tasks not being killed on request

0.8.1 (2018-09-16)
------------------
* :issue:`663`  Interface: updated column headings on home page
* :issue:`660`  Documentation: corrected and improved Linux one-page install
* :issue:`659`  Interface: made the summary tables on the home page sortable by clicking on headers
* :issue:`656`  Install: pegged django-debug-toolbar to 1.9.1 until Django is upgraded
* :issue:`654`  Documentation: supplemented the Orthanc Lua file config option docs
* :issue:`653`  Docs: clarified notes to get link to Orthanc lua file correct on release
* :issue:`652`  Documentation: added docs showing Celery daemonisation in Linux
* :issue:`651`  Documentation: added one-page full setup Ubuntu 18.04 install instructions
* :issue:`650`  Documentation: modified quick install virtualenv docs
* :issue:`649`  Documentation: instructions for updating hosts file for Ubuntu and RabbitMQ
* :issue:`648`  Documentation: clarified Toshiba options when not required
* :issue:`647`  Documentation: updated link to pixelmed
* :issue:`646`  Modified Celery import to avoid name clash in some circumstances
* :issue:`645`  Imports: prevent import failure when text is used in filter thickness field in DX image
* :issue:`644`  Exports: fixed error in exporting non-ASCII CT protocol acquisition names
* :issue:`643`  Installation: updated docs to make use of pip binaries for Postgres connector and numpy, Windows and Linux
* :issue:`642`  Skin dose maps: added catch for error when there are no events in the study
* :issue:`641`  Exports: mammography exports from filtered pages sorted by AGD no longer result in duplicate studies
* :issue:`640`  Exports: error in filter listing for NHSBSP csv exports corrected
* :issue:`639`  Charts: fixed problem where a blank category name may not be displayed correctly
* :issue:`638`  Skin dose maps: added a link to download data for stand-alone openSkin even when map displayed
* :issue:`627`  DICOM Networking: implemented workaround for query "bug" in Impax 6.6
* :issue:`606`  Interface: Made it possible for the user to change his/her password

0.8.0 (2018-06-11)
------------------
* :issue:`635`  Documentation: added Orthanc as preferred third party DICOM Store service
* :issue:`634`  Documentation: updated docs for import and query-retrieve duplicates processing
* :issue:`633`  Charts: fixed issue where charts failed if bar chart series name was null
* :issue:`632`  DICOM: move requests for queries that don't exist now fail gracefully
* :issue:`631`  Skin dose maps: bug fixed that prevented message from displaying on screen when skin dose map cannot be calculated
* :issue:`630`  Documentation: improved installation instructions
* :issue:`628`  Imports: fixed code for importing when there are duplicate DX or MG studies in the database
* :issue:`626`  DICOM: isolated the generate modalities in study function and added testing
* :issue:`625`  Imports: now using event level UIDs to process continued, cumulative and duplicate RDSRs
* :issue:`624`  Charts: removed filter link on number of events histogram as it was not functioning correctly
* :issue:`623`  Imports: changed name of Toshiba image based extractor routine
* :issue:`621`  Documentation: reversed install order of openrem and pynetdicom due to new pydicom release
* :issue:`619`  Documentation: added workaround for outdated dictionary issues
* :issue:`618`  DICOM: fixed image level query that prevented RDSRs from being found
* :issue:`617`  Imports: fixed issue with multi study exams crashing the Toshiba extractor
* :issue:`616`  Documentation: added information for pip download -d
* :issue:`615`  Exports: added Target Exposure Index and Deviation Index to radiographic exports
* :issue:`614`  Exports: handle error when study is deleted during sheet creation for exports
* :issue:`613`  Imports: fixed dual modality type imports after 'dual' designation from ref :issue:`580`
* :issue:`612`  Imports: prevented crash when RDSR was imported with AcquisitionProtocol sequence with no TextValue
* :issue:`610`  DICOM: query-retrieve changed to work for duplicate RDSRs, ref :issue:`114`
* :issue:`609`  Interface: fixed the feature that toggles the selection when clicking anywhere on a display name table row
* :issue:`608`  Interface: fixed the broken sorting of display name table
* :issue:`603`  Interface: fixed JavaScript error if there are any None values in fluoro detail irradiation type table
* :issue:`602`  Skin dose maps: fixed error when there are multiple kVp values for a single irradiation event
* :issue:`599`  Installation: postgres instructions now include note about differing security choices
* :issue:`597`  Skin dose maps: documented that using a production webserver the default timeout value must be increased
* :issue:`596`  Documentation: added docs for using Gunicorn and NGINX on linux
* :issue:`594`  Display: corrected display of dual-plane DAP and RP dose in RF filtered view
* :issue:`593`  Imports: properly handles MultiValue filter material tags and permits aluminium spelling
* :issue:`592`  Documentation: added docs for using IIS on Windows
* :issue:`589`  Exports: now handles zero studies and studies deleted during exports sensibly
* :issue:`587`  Documentation: added instructions for Linux users to rotate logs
* :issue:`586`  Documentation: updated exports and detailed how pulse level data is exported
* :issue:`585`  Documentation: added information about multiple cumulative RDSRs
* :issue:`584`  Import, Interface, Export: RDSR with pulse level data now function
* :issue:`583`  Documentation: added information about dual mode modalities and deleting all from an X-ray unit
* :issue:`582`  Celery: updated results backend as amqp deprecated and slow
* :issue:`581`  Import scripts: interpreter line now always first, functions imported specifically
* :issue:`580`  Imports and Interface: one modality creating both DX and RF can now be handled appropriately
* :issue:`579`  Imports: dummy values for Toshiba CT import function now in settings.py, log file config in docs
* :issue:`578`  Exports: fixed NHSBSP export that was excluding RDSR imported Hologic studies
* :issue:`575`  Exports: export page now updates using AJAX and has a select all button
* :issue:`573`  Exports: corrected and clarified exposure time and duration units, added number of pulses
* :issue:`572`  Interface: homepage now populates as AJAX to increase responsiveness
* :issue:`570`  Charts: simplified chart function code
* :issue:`569`  Charts: fixed frequency issue with mean averages selected
* :issue:`568`  Imports: missing DICOM date-time no longer causes an error
* :issue:`567`  Celery: fixed dual-namespace imports of tasks
* :issue:`566`  Interface: correctly show "assumed patient mass" in case of set value of zero
* :issue:`565`  Interface: correctly handle dose area product with zero value
* :issue:`564`  Skin dose maps: text information on skin dose maps now embedded when saving the 2d or 3d map as a graphic
* :issue:`562`  Skin dose maps: error message on calculation failure now more explicit
* :issue:`561`  Imports: patient orientation modifier now correctly extracted from RDSR
* :issue:`560`  Exports: added study level comments
* :issue:`559`  Interface: date pickers inconsistent start day fixed
* :issue:`558`  Skin dose maps: set defaults instead of crashing if kV, dose, table or tube/detector position are missing
* :issue:`557`  Skin dose maps: improved construction of patient orientation code
* :issue:`556`  Exports: DX exports where TotalNumberOfRadiographicFrames is not populated now export
* :issue:`552`  Documentation: documented extractor for older Toshiba CT scanners
* :issue:`551`  Documentation: added procedure for opening csv files in Excel with non-ASCII characters
* :issue:`550`  Documentation: added a note to describe exposure time and duration for fluoroscopy studies
* :issue:`549`  Documentation: added procedure for fixing laterality on Hologic studies, ref :issue:`411`
* :issue:`547`  Interface: improved handling of available time information for fluoro studies
* :issue:`546`  Query Retrieve: added flag and functionality to query for Toshiba images
* :issue:`544`  Interface: added procedure, requested procedure to summary listings and details and filtering
* :issue:`543`  Interface: added drop-down box to choose how many studies are displayed on filtered pages
* :issue:`542`  Interface: added display name to all detailed html pages
* :issue:`541`  Documentation: updated for celery on Windows
* :issue:`540`  Documentation: updated for current skinDose functionality
* :issue:`539`  Documentation: updated chart document to include series toggle buttons
* :issue:`537`  Charts: hide series function added
* :issue:`536`  Code quality: reduced javascript duplication and collected file groups into subfolders
* :issue:`535`  Interface: fixed problem where category names that included a plus symbol caused filtering and chart issues
* :issue:`534`  Interface: chart drilldown reported as not working - was actually due to a user's database migrations
* :issue:`533`  Query Retrieve: Reduced number of simultaneous associations to one, reused for everything
* :issue:`532`  DICOM: documented how to work-around missing encoding charsets due to old pydicom
* :issue:`529`  Charts: added CT charts of number of irradiation events per study description and requested procedure
* :issue:`528`  Query Retrieve: reduced number of simultaneous associations to one, reused for everything
* :issue:`526`  Code quality: addressed some of the code quality/style issues raised by `Codacy`
* :issue:`525`  Importing: improved mammo import by checking compression force before converting to float
* :issue:`524`  Importing: improved mammo import by checking anode exists before converting to DICOM terms
* :issue:`523`  Importing: changed mammo import to use del_no_match instead of del_mg_im if not mammo
* :issue:`522`  Documentation: made it clearer on offline-install docs that version numbers will change
* :issue:`521`  Testing: added tests for dual source CT imports
* :issue:`520`  Imports: removed XML styling from Philips legacy CT comment creation
* :issue:`519`  Skin dose maps: fixed black on black text issue
* :issue:`518`  Importing: fixed imports where CT Target Region isn't specified
* :issue:`517`  Interface: operator name is now displayed on the detail page for each modality, along with physician for CT and fluoro
* :issue:`516`  Imports: MultiValue person names are now stored as a decoded string, not a list
* :issue:`511`  Testing: develop and other branches can now be deployed to dev.openrem.org and testing.openrem.org automatically
* :issue:`510`  Imports: 'not-patient-indicators' can now be configured in the interface
* :issue:`509`  Skin dose maps: now recalculated on view if recorded height or weight has changed since last calculation
* :issue:`508`  Testing: DX sample files are now tested
* :issue:`507`  Interface: Mammo now filterable by study description, procedure, requested procedure and acquisition protocol
* :issue:`506`  Documentation: updated query-retrieve docs
* :issue:`505`  Charts: n is now displayed on charts
* :issue:`504`  Charts: Fixed issue with null values
* :issue:`503`  Internationalisation: more robust decoding and use of unicode throughout
* :issue:`502`  Testing: tests now work with SQLite3 and PostgreSQL databases
* :issue:`501`  Imports: Changed field type for CodeValue  from 16 chars to text, allows for illegal long values
* :issue:`500`  Imports: Philips SC Dose Info with missing time stamps now import
* :issue:`499`  Imports: Now aborts gracefully with error log if no template in RDSR
* :issue:`498`  Exports: Missing units added to header fields
* :issue:`497`  Interface: Detailed fluoro study view: added irradiation type, pulse rate, dose to ref. point, secondary angle, total DAP and ref. point dose from each irradition type
* :issue:`495`  Charts: Reduced time taken to render scatter plots with multiple series
* :issue:`494`  Charts: Charts now ignore blank and zero-value data when calculating mean, median and number of events
* :issue:`493`  Charts: Added user option to made chart categories all lower case
* :issue:`492`  Exports: Each view is now unique for NHSBSP mammo exports as required by the NCCPM database
* :issue:`491`  Imports, Interface and Exports: CT Dose Check alerts and notifications are now extracted, displayed and exported
* :issue:`490`  Exports: Response object included for messages - removed as now asynchronous
* :issue:`489`  Exports: NHSBSP mammo exports deals with all views, excludes biopsies and specimens
* :issue:`488`  Exports: All exports now include study time
* :issue:`487`  Imports: CT RDSR now imports 'procedure context' correctly
* :issue:`486`  Imports: CT RDSR now imports 'NameOfPhysiciansReadingStudy' correctly
* :issue:`485`  Imports: CT RDSR now imports 'target region' correctly
* :issue:`484`  Exports and Interface: Exports and interface page views are now more efficient and (much) faster
* :issue:`482`  Imports: DX extractor now extracts acquisition protocol, requested procedure name and study name for Fuji Go mobile; extracts acquisition protocol for Toshiba Radrex equipment; extracts requested procedure name from Carestream DRX-Revolution mobiles
* :issue:`480`  Imports: Code and instructions to create and import an RDSR from Toshiba CT dose summary images and studies
* :issue:`476`  Imports: Mixed latin-1 and UTF8 characters now imported, but need to be handled better if possible
* :issue:`475`  Query Retrieve: Made -sr a stand-alone option - it has a very niche use-case!
* :issue:`474`  Logging: Changing to DEBUG logging level in ``local_settings.py`` will now be respected
* :issue:`473`  Query Retrieve: Added tests
* :issue:`472`  Query Retrieve: Overhauled the query retrieve routines
* :issue:`471`  Internationalisation: added configuration and docs to set the timezone
* :issue:`470`  Query Retrieve: Optimised CT filtering
* :issue:`468`  Query Retrieve: Station names can now be used for filtering if returned
* :issue:`467`  Testing: Added tests for mammography RDSR imports
* :issue:`466`  Query Retrieve: RDSR now retrieved in preference to images for MG and DX/CR
* :issue:`465`  Added newer SSDE and water equivalent diameter fields to database
* :issue:`464`  Imports: DX RDSR now imported properly
* :issue:`463`  Imports: Properly checks that Enhanced SR are GE dose reports before importing
* :issue:`460`  Interface: Display names table now sortable
* :issue:`458`  Exports: Filter thicknesses are rounded to max 4 significant figures on export
* :issue:`454`  Exports: Mean filter thickness now reported in exports
* :issue:`453`  Imports: DX with min filter thickness greater than max have values switched on import
* :issue:`452`  Exports: Added CTDIw phantom size to CT exports
* :issue:`451`  Skin dose maps: fixed issue with filters being referenced before being defined
* :issue:`450`  Imports: DX imports with filter thickness of 0.00 are now recorded as such
* :issue:`449`  Exports: Fixed a bug that prevented fluoro exports if protocol names had non-ASCII characters
* :issue:`448`  Documentation: Added a diagram showing the relationship between the OpenREM system components
* :issue:`447`  Imports: Modified rdsr and ctdetail template to import and display data from Pixelmed generated Toshiba RDSR
* :issue:`446`  Import: Extract additional Philips private information for Allura Xper systems, create workaround for missing end angles for rotational acquisitions
* :issue:`445`  Interface: Added function for user to determine between DX and fluoro for ambiguous modalities
* :issue:`444`  Imports: DX systems that submit RDSRs that look like fluoro can now be reclassified using :issue:`445`
* :issue:`443`  Exports: Accession number and ID are now exported to XLSX as text. Thanks to `@LuukO`_
* :issue:`442`  Exports: Fixed RF exports with multiple filters, added tests. Thanks to `@LuukO`_
* :issue:`441`  Charts: Fixed a bug that broke chart links containing non-ASCII characters
* :issue:`440`  Charts: Fixed a bug in sorting.js so that undefined strings are handled correctly
* :issue:`439`  Charts: Added controls for plotting a series per system and calculation histogram data to each filtered view
* :issue:`438`  Skin dose maps: skin dose maps successfully calculated from existing studies; indication of assumed or extracted data shown
* :issue:`434`  Internationalisation: added passing char_set throughout the extractor functions (since largely made redundant again!)
* :issue:`432`  Imports: RDSR import function now looks in comment field for `patient_table_relationship` data
* :issue:`431`  Imports: fixed DX imports with MultiValue filter values (Cu+Al) again!
* :issue:`430`  Exports: fixed DX exports with multiple filters again, added tests
* :issue:`429`  Charts: added new mammo scatter plots. Thanks to `@rijkhorst`_
* :issue:`427`  Testing: added a large number of tests that are automatically run on commit to bitbucket
* :issue:`414`  Reduced use of JavaScript global variables and improved JavaScript objects
* :issue:`411`  Imports: fixed laterality and accumulated AGD failure for Hologic DBT proprietary projection images
* :issue:`323`  Documentation: code autodocumentation largely now working again
* :issue:`318`  Database management: Display names view can be used to review and delete all studies from one source
* :issue:`114`  Imports: Subsequent RDSRs of the same study will now replace existing study in database
* :issue:`61`  Skin dose maps: These have been re-enabled, and currently work for Siemens systems

0.7.4 (2016-10-17)
------------------

* :issue:`436`  Install: temporary fix blocking django-filter latest version that breaks OpenREM
* :issue:`431`  Imports: fixed DX imports with MultiValue filter values (Cu+Al)
* :issue:`430`  Exports: fixed DX exports with multiple filters (Cu + Al)


0.7.3 (2016-08-30)
------------------

* :issue:`426`  Charts: added css so that wide chart data tables are displayed above the filter form div
* :issue:`425`  Exports: fixed error with non-ASCII characters being exported to csv
* :issue:`424`  Charts: fixed error where png or svg export of chart would show incorrect x-axis labels
* :issue:`423`  Charts: fixed error where some chart plotting options were not updated after being changed by the user
* :issue:`422`  Charts: added a button below each chart to toggle the display of the data table
* :issue:`421`  Charts: fixed error where only some scatter plot data was being exported to csv or xls files
* :issue:`420`  Charts: fixed error where frequency pie charts were only showing data from the first system
* :issue:`419`  Interface: fixed error where "Cancel" was ignored when deleting study in Firefox browser
* :issue:`418`  Exports: fixed error when exporting fluoroscopy study with missing xray_filter_material
* :issue:`416`  Charts: improved efficiency of JavaScript
* :issue:`415`  Database: migration for 0.6 upgraded installs to fix acquisition_device_type failures
* :issue:`413`  Documentation: removed erroneous reference to store queue in stop celery command
* :issue:`410`  Charts: fixed display of bar charts containing only one data point
* :issue:`408`  Charts: Increased number of items that can be shown on some Highcharts plots
* :issue:`407`  Fixed issue where skin dose map data was not being calculated on import
* :issue:`406`  Replaced Math.log10 JavaScript function with alternative function to fix IE11 skin dose map error
* :issue:`405`  Altered multi-line cell links in filtered pages so they work with IE8

0.7.1 (2016-06-10)
------------------

* :issue:`403`  Now deals with PersonName fields with latin-1 extended characters correctly
* :issue:`402`  Skin dose map data pickle files saved using gzip compression to save space
* :issue:`401`  Updated skin dose map documentation to say it won't be in this release
* :issue:`400`  Strings are encoded as UTF-8 before being hashed to prevent errors with non-ASCII characters
* :issue:`399`  Migration file brought up to date for 0.6 to 0.7 upgrades
* :issue:`398`  Skin exposure maps are now stored in folders (feature postponed for future release)
* :issue:`397`  Skin exposure maps no longer available until orientation errors are fixed
* :issue:`396`  Charts: zooming on bar charts of average value vs. category now works
* :issue:`395`  Docs: offline Windows install instructions created, plus offline upgrade instructions
* :issue:`394`  Charts: made charts resize to fit containing div when browser is resized
* :issue:`392`  Charts: normalised histogram tooltip now correctly reports frequency
* :issue:`391`  Basic troubleshooting is now documented
* :issue:`390`  Charts: mammography and fluoroscopy charts added
* :issue:`389`  Charts: series without a name are now plotted under the name of `Blank` rather than not being plotted at all
* :issue:`387`  Added laterality to mammography exports
* :issue:`385`  Fixed issue with non-ASCII letters in RDSR sequence TextValue fields
* :issue:`384`  Fluoro exports for OpenSkin only consider copper filters now
* :issue:`383`  Refreshed settings.py to django 1.8 including updating template settings and TEMPLATE_CONTEXT_PROCESSORS
* :issue:`380`  Tube current now extracted from Siemens Intevo RDSR despite non-conformance
* :issue:`379`  Exposure time now populated for fluoro if not supplied by RDSR
* :issue:`378`  The display name of multiple systems can now be updated together using a single new name
* :issue:`376`  Corrected an ill-advised model change
* :issue:`374`  CTDIw phantom size now displayed in CT detail view
* :issue:`373`  Charts in some releases used GT rather than greater than or equal to for start date, now fixed
* :issue:`372`  Mammography studies now record an accumulated AGD per breast. Existing joint accumulated AGD values won't be
  changed. Ordering by Accumulated AGD now creates an entry per accumulated AGD, one per breast
* :issue:`371`  Mammo RDSR generates average mA where not recorded, mammo image populates mA
* :issue:`370`  Added study description to mammography export
* :issue:`369`  Bi-plane fluoroscopy studies now export correctly
* :issue:`368`  Mammo RDSR now imports correctly
* :issue:`365`  Tube filtration is now displayed in the RF detail view
* :issue:`364`  Philips Allura fluorscopy RDSRs now import correctly
* :issue:`362`  Display of RF where bi-plane RDSRs have been imported no longer crash the interface
* :issue:`360`  Charts: saving data from average data charts as csv or xls now includes frequency values
* :issue:`359`  Added missing 'y' to query retrieve command line help
* :issue:`358`  Charts: chart sorting links and instructions now hidden when viewing histograms
* :issue:`357`  Charts: button to return from histogram now displays the name of the main chart
* :issue:`356`  Charts: histogram normalise button appears for all appropriate charts
* :issue:`355`  Charts: sorting now works as expected for plots with a series per system
* :issue:`352`  Fixed CT xlsx exports that had complete study data in each series protocol sheet (from earlier beta)
* :issue:`351`  Charts: simplified chart JavaScript and Python code
* :issue:`350`  DICOM networking documented for use with 3rd party store and advanced use with native
* :issue:`348`  Study delete confirmation page now displays total DAP for DX or CR radiographic studies
* :issue:`346`  Charts: exporting a chart as an image no longer requires an internet connection
* :issue:`345`  CSV size imports in cm are now stored as m in the database. Interface display of size corrected.
* :issue:`343`  Charts: user can now specify number of histogram bins in the range of 2 to 40
* :issue:`342`  Charts: improved the colours used for plotting chart data
* :issue:`340`  Fixed store failure to save due to illegal values in Philips private tags, improved exception code
* :issue:`339`  Improved extraction of requested procedure information for radiographic studies
* :issue:`338`  Fix Kodak illegally using comma in filter thickness values
* :issue:`335`  DICOM Store keep_alive and echo_scu functions now log correctly
* :issue:`334`  Fixed issue with tasks needing to be explicitly named
* :issue:`333`  Fixed StoreSCP not starting in beta 11 error
* :issue:`332`  Charts: some charts can now be plotted with a series per x-ray system
* :issue:`331`  Keep_alive tasks are now discarded if not executed, so don't pile up
* :issue:`329`  All existing logging is now done via the same log files
* :issue:`328`  Store SCP no longer uses Celery tasks
* :issue:`327`  Celery workers now only take one task at a time
* :issue:`325`  Charts: switching charts off now leaves the user on the same page, rather than going to the home page
* :issue:`324`  Charts: forced chart tooltip background to be opaque to make reading the text easier
* :issue:`320`  The week now begins on Monday rather than Sunday on date form fields
* :issue:`316`  Query retrieve function can now exclude and include based on strings entered
* :issue:`315`  Charts: made size of exported chart graphics follow the browser window size
* :issue:`314`  One version number declaration now used for distribute, docs and interface
* :issue:`313`  Replaced non-working function with code to extract SeriesDescription etc in query response message
* :issue:`312`  Display names are now grouped by modality
* :issue:`311`  Queries are deleted from database after a successful C-Move
* :issue:`310`  Series level QR feedback now presented. Any further would require improvements in pynetdicom
* :issue:`309`  StoreSCP now deals safely with incoming files with additional transfer syntax tag
* :issue:`308`  Secondary capture images that don't have the manufacturer field no longer crash the StoreSCP function
* :issue:`306`  Charts: added a button to each chart to toggle full-screen display
* :issue:`305`  Added links to documentation throughout the web interface
* :issue:`304`  Date of birth is now included in all exports that have either patient name or ID included
* :issue:`303`  Fixed a typo in 0.6.0 documents relating to the storescp command
* :issue:`302`  Improved handling of Philips Dose Info objects when series information sequence has UN value representation
* :issue:`301`  Charts: fixed bug that could stop average kVp and mAs radiographic plots from working
* :issue:`300`  Calling AE Title for Query Retrieve SCU is now configured not hardcoded
* :issue:`299`  Hash of MultiValued DICOM elements now works
* :issue:`298`  Added ordering by accumulated AGD for mammographic studies
* :issue:`297`  Fixed ordering by Total DAP for radiographic studies
* :issue:`296`  StoreSCP now logs an error message and continues if incoming file has problems
* :issue:`295`  Charts: fixed bug that arose on non-PostgreSQL databases
* :issue:`294`  Harmonised time display between filter list and detail view, both to HH:mm
* :issue:`292`  Added keep-alive and auto-start to DICOM stores
* :issue:`291`  Charts: fixed issue with CTDI and DLP not showing correct drilldown data
* :issue:`290`  Added new tables and fields to migration file, uses :issue:`288` and median code from :issue:`241`
* :issue:`289`  Crispy forms added into the requires file
* :issue:`288`  Added device name hashes to migration file
* :issue:`286`  Increased granularity of permission groups
* :issue:`285`  Tidied up Options and Admin menus
* :issue:`284`  Fixed DICOM Query that looped if SCP respected ModalitiesInStudy
* :issue:`282`  Missing javascript file required for IE8 and below added
* :issue:`281`  Added check to import function to prevent extract failure
* :issue:`280`  Fixed typo in mammography export
* :issue:`279`  Charts: Fixed issue with median CTDI series from appearing
* :issue:`278`  Charts: Fixed javascript namespace pollution that caused links to fail
* :issue:`277`  Overhaul of acquisition level filters to get tooltip generated filters to follow through to export
* :issue:`276`  Unique fields cannot have unlimited length in MySQL - replaced with hash
* :issue:`274`  Charts: Fixed legend display issue
* :issue:`273`  Charts: Added plots of average kVp and mAs over time for DX
* :issue:`272`  Tweak to display of exam description for DX
* :issue:`271`  Fixed DX import failure where ``AcquisitionDate`` or ``AcquisitionTime`` are ``None``
* :issue:`270`  Django 1.8 Admin site has a 'view site' link. Pointed it back to OpenREM
* :issue:`268`  Improved population of procedure_code_meaning for DX imports
* :issue:`266`  DICOM C-Store script added back in - largely redundant with web interface
* :issue:`265`  DICOM Store and Query Retrieve services documented
* :issue:`263`  Settings for keeping or deleting files once processed moved to database and web interface
* :issue:`262`  Dealt with issue where two exposures from the same study would race on import
* :issue:`260`  Fixed issue where import and export jobs would get stuck behind StoreSCP task in queue
* :issue:`259`  Link to manage users added to Admin menu
* :issue:`258`  Fixed DX import error where manufacturer or model name was not provided
* :issue:`257`  Documentation update
* :issue:`256`  Fixed errors with non-ASCII characters in imports and query-retrieve
* :issue:`255`  Charts: Small y-axis values on histograms are more visible when viewing full-screen
* :issue:`254`  Charts: Simplified chart data processing in the templates
* :issue:`253`  Charts: AJAX used to make pages responsive with large datasets when charts enabled
* :issue:`252`  Fixed duplicate entries in DX filtered data for studies with multiple exposures
* :issue:`248`  Charts: can now be ordered by frequency or alphabetically
* :issue:`247`  Fixed incorrect reference to manufacturer_model_name
* :issue:`246`  Charts: Added median data for PostgreSQL users
* :issue:`245`  Fixed error in csv DX export
* :issue:`244`  Fixed issue where scripts wouldn't function after upgrade to Django 1.8
* :issue:`243`  Added distance related data to DX exports
* :issue:`242`  Distance source to patient now extracted from DX images
* :issue:`241`  Charts: Median values can be plotted for PostgreSQL users
* :issue:`240`  Charts: Improved DAP over time calculations
* :issue:`239`  Configurable equipment names to fix multiple sources with the same station name
* :issue:`237`  Charts: Tidied up plot data calculations in ``views.py``
* :issue:`235`  Added patient sex to each of the exports
* :issue:`234`  Charts: Fixed error with datetime combine
* :issue:`232`  Charts: on or off displayed on the home page
* :issue:`231`  Charts: made links from requested procedure frequency plot respect the other filters
* :issue:`230`  Fixed error in OperatorsName field in DICOM extraction
* :issue:`229`  Charts: Added chart of DLP per requested procedure
* :issue:`223`  Charts: speed improvement for weekday charts
* :issue:`217`  Charts: Further code optimisation to speed up calculation time
* :issue:`207`  DICOM QR SCU now available from web interface
* :issue:`206`  DICOM Store SCP configuration now available from web interface
* :issue:`183`  Added options to store patient name and ID, and options to hash name, ID and accession number
* :issue:`171`  Root URL now resolves so ``/openrem`` is not necessary
* :issue:`151`  Suspected non-patient studies can now be filtered out
* :issue:`135`  GE Senographe DS now correctly records compression force in Newtons for new imports
* :issue:`120`  Improved testing of data existing for exports
* :issue:`118`  Upgraded to Django 1.8
* :issue:`70`   User is returned to the filtered view after deleting a study
* :issue:`61`   Skin dose maps for fluoroscopy systems can now be calculated and displayed

0.6.2 (2016-01-27)
------------------
* :issue:`347`  Django-filter v0.12 has minimum Django version of 1.8, fixed OpenREM 0.6.2 to max django-filter 0.11
* :issue:`341`  Changed references to the OpenSkin repository for 0.6 series.

0.6.1 (2015-10-30)
------------------
* :issue:`303`  Corrected name of Store SCP command in docs

0.6.0 (2015-05-14)
------------------

* :issue:`227`  Fixed import of RDSRs from Toshiba Cath Labs
* :issue:`226`  Charts: Updated Highcharts code and partially fixed issues with CTDIvol and DLP combined chart
* :issue:`225`  Charts: Added link from mAs and kVp histograms to associated data
* :issue:`224`  Charts: Added link from CTDIvol histograms to associated data
* :issue:`221`  Charts: Fixed issue where filters at acquisition event level were not adequately restricting the chart data
* :issue:`219`  Charts: Fixed issue where some charts showed data beyond the current filter
* :issue:`217`  Charts: Code optimised to speed up calculation time
* :issue:`216`  Fixed typo that prevented import of RSDR when DICOM store settings not present
* :issue:`215`  Charts: Fixed x-axis labels for mean dose over time charts
* :issue:`214`  Charts: Improved consistency of axis labels
* :issue:`213`  Fixed admin menu not working
* :issue:`212`  Charts: Created off-switch for charts
* :issue:`210`  OpenSkin exports documented
* :issue:`209`  Charts: Fixed server error when CT plots switched off and filter form submitted
* :issue:`208`  Charts: Fixed blank chart plotting options when clicking on histogram tooltip link
* :issue:`205`  Charts: Fixed issue of histogram tooltip links to data not working
* :issue:`204`  Charts: Fixed issue of not being able to export with the charts features added
* :issue:`203`  Charts: Fixed display of HTML in plots issue
* :issue:`202`  Charts: Added mean CTDIvol to charts
* :issue:`200`  Charts: Now exclude Philips Ingenuity SPRs from plots
* :issue:`196`  Added comments and entrance exposure data to DX export
* :issue:`195`  Fixed error with no users on fresh install
* :issue:`194`  Added more robust extraction of series description from DX
* :issue:`193`  Charts: Fixed reset of filters when moving between pages
* :issue:`192`  Created RF export for OpenSkin
* :issue:`191`  Charts: Factored out the javascript from the filtered.html files
* :issue:`190`  Charts: Added time period configuration to dose over time plots
* :issue:`189`  Charts: Fixed plotting of mean doses over time when frequency not plotted
* :issue:`187`  Charts: Merged the charts work into the main develop branch
* :issue:`186`  Fixed duplicate data in DX exports
* :issue:`179`  Charts: Added kVp and mAs plots for DX
* :issue:`177`  Charts: Fixed issue with date ranges for DX mean dose over time charts
* :issue:`176`  Charts: Added link to filtered dataset from mean dose over time charts
* :issue:`175`  Charts: Allowed configuration of the time period for mean dose trend charts to improve performance
* :issue:`174`  Charts: Fixed number of decimal places for mean DLP values
* :issue:`173`  Charts: Fixed plot of mean DLP over time y-axis issue
* :issue:`170`  Charts: Added plot of mean dose over time
* :issue:`169`  Charts: Improved chart colours
* :issue:`157`  Charts: Added chart showing number of studies per day of the week, then hour in the day
* :issue:`156`  Charts: Fixed issue with some protocols not being displayed
* :issue:`155`  Charts: Added chart showing relative frequency of protocols and study types
* :issue:`140`  Charts: Added configuration options
* :issue:`139`  Charts: Link to filtered dataset from histogram chart
* :issue:`138`  Charts: Number of datapoints displayed on tooltip
* :issue:`135`  Mammography compression force now only divides by 10 if model contains *senograph ds* **Change in behaviour**
* :issue:`133`  Documented installation of NumPy, initially for charts
* :issue:`41`   Preview of DICOM Store SCP now available
* :issue:`20`   Modality sections are now suppressed until populated


0.5.1 (2015-03-12)
------------------

* :issue:`184`  Documentation for 0.5.1
* :issue:`180`  Rename all reverse lookups as a result of :issue:`62`
* :issue:`178`  Added documentation regarding backing up and restoring PostgreSQL OpenREM databases
* :issue:`172`  Revert all changes made to database so :issue:`62` could take place first
* :issue:`165`  Extract height and weight from DX, height from RDSR, all if available
* :issue:`161`  Views and exports now look for accumulated data in the right table after changes in :issue:`159` and :issue:`160`
* :issue:`160`  Created the data migration to move all the DX accumulated data from TID 10004 to TID 10007
* :issue:`159`  Modified the DX import to populate TID 10007 rather than TID 10004. RDSR RF already populates both
* :issue:`158`  Demo website created by DJ Platten: http://demo.openrem.org/openrem
* :issue:`154`  Various decimal fields are defined with too few decimal places - all have now been extended.
* :issue:`153`  Changed home page and modality pages to have whole row clickable and highlighted
* :issue:`150`  DJ Platten has added Conquest configuration information
* :issue:`137`  Carestream DX multiple filter thickness values in a DS VR now extracted correctly
* :issue:`113`  Fixed and improved recording of grid information for mammo and DX and RDSR import routines
* :issue:`62`   Refactored all model names to be less than 39 characters and be in CamelCase to allow database migrations and
  to come into line with PEP 8 naming conventions for classes.


0.5.0 (2014-11-19)
------------------

* Pull request from DJ Platten: Improved display of DX data and improved export of DX data
* :issue:`132`  Fixed mammo export error that slipped in before the first beta
* :issue:`130`  Only creates ExposureInuAs from Exposure if Exposure exists now
* :issue:`128`  Updated some non-core documentation that didn't have the new local_settings.py reference or the new
  openremproject folder name
* :issue:`127`  DX IOD studies with image view populated failed to export due to lack of conversion to string
* :issue:`126`  Documentation created for the radiographic functionality
* :issue:`125`  Fixes issue where Hologic tomo projection objects were dropped as they have the same event time as the 2D element
* :issue:`123`  Fixed issue where filters came through on export as lists rather than strings on some installs
* :issue:`122`  Exports of RF data should now be more useful when exporting to xlsx. Will need refinement in the future
* :issue:`26`   Extractors created for radiographic DICOM images. Contributed by DJ Platten
* :issue:`25`   Views and templates added for radiographic exposures - either from RDSRs or from images - see :issue:`26`.
  Contributed by DJ Platten
* :issue:`9`    Import of \*.dcm should now be available from Windows and Linux alike


0.4.3 (2014-10-01)
------------------

* :issue:`119`  Fixed issue where Celery didn't work on Windows. Django project folder is now called openremproject instead of openrem
* :issue:`117`  Added Windows line endings to patient size import logs
* :issue:`113`  Fixed units spelling error in patient size import logs
* :issue:`112`  File system errors during imports and exports are now handled properly with tasks listed in error states on the summary pages
* :issue:`111`  Added abort function to patient size imports and study exports
* :issue:`110`  Converted exports to use the FileField handling for storage and access, plus modified folder structure.
* :issue:`109`  Added example ``MEDIA_ROOT`` path for Windows to the install docs
* :issue:`108`  Documented ownership issues between the webserver and Celery
* :issue:`107`  Documented process for upgrading to 0.4.2 before 0.4.3 for versions 0.3.9 or earlier
* :issue:`106`  Added the duration of export time to the exports table. Also added template formatting tag to convert seconds to natural time
* :issue:`105`  Fixed bug in Philips CT import where :py:class:`decimal.Decimal` was not imported before being used in the age calculation
* :issue:`104`  Added documentation for the additional study export functions as a result of using Celery tasks in task :issue:`19` as well as documentation for the code
* :issue:`103`  Added documentation for using the web import of patient size information as well as the new code
* :issue:`102`  Improved handling of attempts to process patient size files that have been deleted for when users go back in the browser after the process is finished
* :issue:`101`  Set the security of the new patient size imports to prevent users below admin level from using it
* :issue:`100`  Logging information for patient size imports was being written to the database - changed to write to file
* :issue:`99`   Method for importing remapp from scripts and for setting the `DJANGO_SETTINGS_MODULE` made more robust so that it should work out of the box on Windows, debian derivatives and virtualenvs
* :issue:`98`   Versions 0.4.0 to 0.4.2 had a settings.py.new file to avoid overwriting settings files on upgrades; renaming this file was missing from the installation documentation for new installs
* :issue:`97`   Changed the name of the export views file from ajaxviews as ajax wasn't used in the end
* :issue:`96`   Changed mammo and fluoro filters to use named fields to avoid needing to use the full database path
* :issue:`93`   Set the security of the new exports to prevent users below export level from creating or downloading exports
* :issue:`92`   Add `NHSBSP specific mammography csv export` from Jonathan Cole - with Celery
* :issue:`91`   Added documentation for Celery and RabbitMQ
* :issue:`90`   Added delete function for exports
* :issue:`89`   Added the Exports navigation item to all templates, limited to export or admin users
* :issue:`88`   Converted fluoroscopy objects to using the Celery task manager after starting with CT for :issue:`19`
* :issue:`87`   Converted mammography objects to using the Celery task manager after starting with CT for :issue:`19`
* :issue:`86`   Digital Breast Tomosynthesis systems have a projections object that for Hologic contains required dosimetry information
* :issue:`85`   Fix for bug introduced in :issue:`75` where adaption of ptsize import for procedure import broke ptsize imports
* :issue:`74`   'Time since last study' is now correct when daylight saving time kicks in
* :issue:`39`   Debug mode now defaults to False
* :issue:`21`   Height and weight data can now be imported through forms in the web interface
* :issue:`19`   Exports are now sent to a task manager instead of locking up the web interface

Reopened issue
``````````````

* :issue:`9`    Issue tracking import using \*.dcm style wildcards reopened as Windows ``cmd.exe`` shell doesn't do wildcard expansion, so this will need to be handled by OpenREM in a future version

0.4.2 (2014-04-15)
------------------

* :issue:`83`   Fix for bug introduced in :issue:`73` that prevents the import scripts from working.

0.4.1 (2014-04-15)
------------------

* :issue:`82`   Added instructions for adding users to the release notes

0.4.0 (2014-04-15)
------------------

..  note::

    * :issue:`64` includes **changes to the database schema and needs a user response** - see `version 0.4.0 release notes <https://docs.openrem.org/page/release-0.4.0.html>`_
    * :issue:`65` includes changes to the settings file which **require settings information to be copied** and files moved/renamed - see `version 0.4.0 release notes <https://docs.openrem.org/page/release-0.4.0.html>`_


* :issue:`80`   Added docs for installing Apache with auto-start on Windows Server 2012. Contributed by JA Cole
* :issue:`79`   Updated README.rst instructions
* :issue:`78`   Moved upgrade documentation into the release notes page
* :issue:`77`   Removed docs builds from repository
* :issue:`76`   Fixed crash if exporting from development environment
* :issue:`75`   Fixed bug where requested procedure wasn't being captured on one modality
* :issue:`73`   Made launch scripts and ptsizecsv2db more robust
* :issue:`72`   Moved the secret key into the local documentation and added instructions to change it to release notes and install instructions
* :issue:`71`   Added information about configuring users to the install documentation
* :issue:`69`   Added documentation about the new delete study function
* :issue:`68`   Now checks sequence code meaning and value exists before assigning them. Thanks to JA Cole
* :issue:`67`   Added 'Contributing authors' section of documentation
* :issue:`66`   Added 'Release notes' section of documentation, incuding this file
* :issue:`65`   Added new ``local_settings.py`` file for database settings and other local settings
* :issue:`64`   Fixed imports failing due to non-conforming strings that were too long
* :issue:`63`   The mammography import code stored the date of birth unnecessarily. Also now gets decimal_age from age field if necessary
* :issue:`60`   Removed extraneous colon from interface data field
* :issue:`18`   Studies can now be deleted from the web interface with the correct login
* :issue:`16`   Added user authentication with different levels of access
* :issue:`9`    Enable import of ``*.dcm``


0.3.9 (2014-03-08)
------------------
..  note:: :issue:`51` includes changes to the database schema -- make sure South is in use before upgrading. See https://docs.openrem.org/page/upgrade.html

* :issue:`59`   CSS stylesheet referenced particular fonts that are not in the distribution -- references removed
* :issue:`58`   Export to xlsx more robust - limitation of 31 characters for sheet names now enforced
* :issue:`57`   Modified the docs slightly to include notice to convert to South before upgrading
* :issue:`56`   Corrected the mammography target and filter options added for issue :issue:`44`
* :issue:`53`   Dates can now be selected from a date picker widget for filtering studies
* :issue:`52`   Split the date field into two so either, both or neither can be specified
* :issue:`51`   Remove import modifications from issue :issue:`28` and :issue:`43` now that exports are filtered in a better way after :issue:`48` and :issue:`49` changes.
* :issue:`50`   No longer necessary to apply a filter before exporting -- docs changed to reflect this
* :issue:`49`   CSV exports changed to use the same filtering routine introduced for :issue:`48` to better handle missing attributes
* :issue:`48`   New feature -- can now filter by patient age. Improved export to xlsx to better handle missing attributes
* :issue:`47`   Install was failing on pydicom -- fixed upstream

0.3.8 (2014-03-05)
------------------

* --    File layout modified to conform to norms
* :issue:`46`   Updated documentation to reflect limited testing of mammo import on additional modalities
* :issue:`45`   mam.py was missing the licence header - fixed
* :issue:`44`   Added Tungsten, Silver and Aluminum to mammo target/filter strings to match -- thanks to DJ Platten for strings
* :issue:`43`   Mammography and Philips CT import and export now more robust for images with missing information such as accession number and collimated field size
* :issue:`42`   Documentation updated to reflect :issue:`37`
* :issue:`37`   Studies now sort by time and date


0.3.7 (2014-02-25)
------------------

* :issue:`40`   Restyled the filter section in the web interface and added a title to that section
* :issue:`38`   Column titles tidied up in Excel exports
* :issue:`36`   openrem_ptsizecsv output of log now depends on verbose flag
* :issue:`35`   Numbers no longer stored as text in Excel exports

0.3.6 (2014-02-24)
------------------

* :issue:`34`   Localised scripts that were on remote web servers in default Bootstrap code
* :issue:`33`   Documentation now exists for adding data via csv file
* :issue:`24`   Web interface has been upgraded to Bootstrap v3
* :issue:`5`    Web interface and export function now have some documentation with screenshots


0.3.5-rc2 (2014-02-17)
----------------------

* :issue:`32`   Missing sys import bug prevented new patient size import from working

0.3.5 (2014-02-17)
------------------

* --    Prettified this document!
* :issue:`31`   Promoted patient size import from csv function to the scripts folder so it will install and can be called from the path
* :issue:`30`   Improved patient size import from csv to allow for arbitary column titles and study instance UID in addition to accession number.
* :issue:`29`   Corrected the docs URL in the readme

0.3.4-rc2 (2014-02-14)
----------------------

* :issue:`28`   XLSX export crashed if any of the filter fields were missing. Now fills on import with 'None'
* :issue:`27`   Use requested procedure description if requested procedure code description is missing


0.3.4 (2014-02-14)
------------------

* --    General improvements and addition of logo to docs
* :issue:`23`   Added Windows XP MySQL backup guide to docs
* :issue:`22`   Added running Conquest as a Windows XP service to docs
* :issue:`15`   Added version number and copyright information to xlsx exports
* :issue:`14`   Added version number to the web interface
* :issue:`13`   Improve the docs with respect to South database migrations


0.3.3-r2 (2014-02-04)
---------------------

* :issue:`12`   Added this version history
* :issue:`11`   Documentation is no longer included in the tar.gz install file -- see http://openrem.trfd.org instead

0.3.3 (2014-02-01)
------------------

..      Note::

        Installs of OpenREM earlier than 0.3.3 will break on upgrade if the scripts are called from other programs.
        For example openrem_rdsr is now called openrem_rdsr.py

* --    Added warning of upgrade breaking existing installs to docs
* :issue:`10`   Added .py suffix to the scripts to allow them to be executed on Windows (thanks to DJ Platten)
* :issue:`8`    Removed superfluous '/' in base html file, harmless on linux, prevented Windows loading stylesheets (thanks to DJ Platten)
* :issue:`7`    Added windows and linux path examples for test SQLite database creation
* :issue:`6`    Corrected renaming of example files installation instruction (thanks to DJ Platten)
* :issue:`4`    Added some text to the documentation relating to importing files to OpenREM
* :issue:`3`    Corrected copyright notice in documentation


0.3.2 (2014-01-29)
------------------

*       Initial version uploaded to bitbucket.org


..  _`NHSBSP specific mammography csv export`: https://bitbucket.org/jacole/openrem-visualisation/commits/0ee416511c847960523a6475ef33ac72#comment-1003330
..  _@rijkhorst: https://bitbucket.org/rijkhorst/
..  _@LuukO: https://bitbucket.org/LuukO/
..  _Codacy: https://www.codacy.com/app/OpenREM/openrem
