Quickstart: Transcribe a song
First, ensure you are running Linux and have Docker installed. Then, run this one time setup command, which will download a ~4GB Docker container and ~100MB of data to a cache directory (~/.sheetsage by default).

ROOT=https://raw.githubusercontent.com/aidan-luu/sheetsage/main; wget $ROOT/prepare.sh && wget $ROOT/sheetsage.sh && chmod +x *.sh && ./prepare.sh
Once this setup completes, transcribing a song is as simple as running:

./sheetsage.sh https://www.youtube.com/watch?v=fHI8X4OXluQ

This will create a directory output/<UUID> containing a PDF with the lead sheet (along with the corresponding LilyPond file), and a MIDI file containing audio-aligned melody and harmony.

You can also run Sheet Sage on a local file:

./sheetsage.sh my_song.mp3

Improving results
Hopefully, the above command will produce reasonable results for most Western music (especially pop) without additional configuration. If the results are unsatisfactory, there are a few things you can try.

Transcribing a shorter segment
Sheet Sage was trained on short (~24 second) segments of music. If you only want to transcribe a segment of that duration or less, you will likely get better results by specifying the starting timestamp of that segment. For example, try:

./sheetsage.sh -s 17 --legacy_behavior <YOUR_SONG>

This will configure Sheet Sage to transcribe a 24s segment from the detected downbeat closest to 17s. Hopefully, the results will be improved over that same segment within the context of the full song.

Fixing tempo issues
Sheet Sage runs beat detection on your input so that it can output a reasonable sheet music score. Occasionally, the beat detection system will detect the tempo as twice or half as fast as the actual tempo, leading to quarter notes rendering as half notes or eighth notes. To fix this, first estimate the tempo (no need to be super precise) and then pass this estimate to Sheet Sage:

./sheetsage.sh --beats_per_minute_hint 120 <YOUR_SONG>

Fixing downbeat issues
In order to draw bar lines in the output, Sheet Sage must detect not only beats but also downbeats. Thanks to madmom, the detection of the former is quite good, but the latter can be brittle, leading to poor transcriptions. If Sheet Sage is detecting the wrong beats as downbeats for your song, you can override the automatic detection by forcing Sheet Sage to interpret your input segment timestamps as downbeats, which should improve transcription quality:

./sheetsage.sh -s 17.0 -e 40.0 --segment_hints_are_downbeats <YOUR_SONG>

If the downbeats are still incorrect, try nudging your timestamps until the correct downbeat is selected.

Using Jukebox for general quality improvements (GPU required)
Our paper demonstrates that using features from OpenAI Jukebox can improve transcription performance. To enable this feature, you must first ensure you have a GPU w/ at least 12GB memory and CUDA installed (nvidia-smi should list your GPU). Then, run ./prepare.sh -j which will download ~10GB of additional files (most of this is the Jukebox model) to ~/.sheetsage. Then, run:

./sheetsage.sh -j <YOUR_SONG>

Note that this will likely take several minutes to complete - consider transcribing a shorter segment when using Jukebox (see above)

HookTheory dataset
Sheet Sage was trained on a new dataset of 50 hours of aligned melody and harmony annotations derived from Hooktheory's TheoryTab DB, which we release alongside this system under a CC BY-NC-SA 3.0 license. The dataset can be downloaded here as a simple, MIR-friendly JSON format (20MB) (no audio is included). Click here for a standalone IPython notebook demonstrating how to explore the dataset.

The dataset is a simple JSON object where each annotation is keyed by its HookTheory ID. We pre-split the data (see the split field) into train, validation, and testing subsets in a 8:1:1 ratio stratified by artist name. The tags field contains various high-level tags; for training melody transcription models, we recommend filtering down to annotations that contain the AUDIO_AVAILABLE and MELODY tags, and filtering out annotations that contain the TEMPO_CHANGES tag. In the alignment field, we include both the original user-specified alignment from HookTheory and our refined alignment (see our paper for details); your system may use either (or neither!) during training.

To simplify evaluation and decouple the melody transcription task from our internal format, we also release the test set annotations as standard audio-aligned MIDI files using our refined alignments: Hooktheory_Test_MIDI.tar.gz. To evaluate a new melody transcription system, use only the information available in Hooktheory_Test_Segments.json as input (audio identifier and, optionally, segment boundaries) and have your system output a MIDI file (named using the dataset key) to a directory. Then, run python -m sheetsage.eval Hooktheory_Test_MIDI.tar.gz <MY_OUTPUT_DIR> --allow_abstain to evaluate your system. Along with the evaluation metrics, we recommend reporting the percentage of annotations you were able to evaluate on at time of publication, and evaluating all models you are comparing on the same set of audio files.

Note that, as our primary focus of this work was on melody transcription, we do not provide any official recommendations for how to evaluate chord recognition models on this HookTheory data, though we strongly encourage followup work to establish standards for doing so!

A full list of all of the files we release as part of this dataset is as follows (bolded files constitute the "canonical" dataset we release as part of our paper, non-bolded files are auxiliary):

Hooktheory.json.gz, full dataset as simplified, MIR-friendly JSON
Hooktheory_Test_Segments.json, system inputs for official evaluation
Hooktheory_Test_MIDI.tar.gz, reference outputs for official evaluation
Hooktheory_Raw.json.gz, full dataset as raw, proprietary functional format from HookTheory.
Hooktheory_Train_Segments.json, training inputs in official evaluation format
Hooktheory_Train_MIDI.tar.gz, training targets in official evaluation format
Hooktheory_Valid_Segments.json, validation inputs in official evaluation format
Hooktheory_Valid_MIDI.tar.gz, validation targets in official evaluation format
Development and advanced usage
To get started with development, start by running:

git clone git@github.com:aidan-luu/sheetsage.git --single-branch

Because Sheet Sage has a considerable number of (fairly brittle) dependencies, development through Docker is strongly recommended. To do so, navigate to the docker directory and run the ./run.sh script, which will launch a development container in the background (any changes to the library on the host will propagate to the container). Then run ./shell.sh to tunnel into the container.

Once in the container, try running python -m unittest discover -v to run the test cases. You may also need to run python -m sheetsage.assets to download all development assets.

Transcription with Jupyter
Especially when using Jukebox, it can be convenient to run transcription in a Jupyter notebook (to avoid setting up the models every time you transcribe a new song). To do so, navigate to the docker directory and run ./run.sh followed by ./notebook.sh. Open one of the displayed links (should include a token parameter) in your browser, and interact with the Inference.ipynb file.

Licensing considerations
While all of the code in this repository is released under a permissive MIT license, the "Sheet Sage" system as a whole contains numerous additional licensing considerations that especially affect commercial use.

Because they are trained on user contributions to HookTheory, the transcription models underneath the hood of Sheet Sage (specifically, all of the files referenced in sheetsage.json which are downloaded by the prepare.sh script) are released under CC BY-NC-SA 3.0. Moreover, Sheet Sage makes use of madmom, Jukebox, and Melisma, which all have additional licensing terms affecting commercial use.

Please ensure your use of Sheet Sage complies with all licensing terms.
