# MObyGaze Dataframe Column Explanation

Sample rows from the dataframe: 

| clip_index | movie_id | annotator   | label         | concepts                            | comment                                    | start_frame | end_frame | imdb_key  | movie_title                          | framerate | text                                                                 | label_speech | start_clip | end_clip | label_audio |
|------------|----------|-------------|---------------|-------------------------------------|--------------------------------------------|-------------|-----------|-----------|-------------------------------------|-----------|---------------------------------------------------------------------|--------------|------------|----------|-------------|
| id_0       | 48       | annotator_1 | Easy Negative | []                                  | ['credits']                                | 0           | 2785      | tt0097576 | Indiana Jones and the Last Crusade  | 23.976    |                                                                     | 0            | 0          | 174      | 0           |
| id_1       | 48       | annotator_1 | Easy Negative | []                                  | ['credits minimal dialogue']               | 2786        | 3567      | tt0097576 | Indiana Jones and the Last Crusade  | 23.976    | Dismount! Herman's horsesick! Chaps, no one wander off. Some of the passageways in here can run for miles. | 0            | 174        | 222      | 0           |
| id_2       | 48       | annotator_1 | Easy Negative | []                                  | ['young indy, nothing to annotate']        | 3568        | 17017     | tt0097576 | Indiana Jones and the Last Crusade  | 23.976    | "I don't think this is such a good idea. What is it? Alfred, ..... [subtitle continues]" | 0            | 223        | 1063     | 0           |


* `clip_index`:represents the index of individual annotations made by an annotator for each movie.
* `movie_id`: unique identifier for each movie, serving as the primary key in the database for indexing purposes.
* `annotator`
* `label`: annotation choices as described in the [Objectification Thesaurus](https://github.com/husky-helen/MObyGaze/blob/main/dataset/objectification-thesaurus.json)
* `concepts`: array of concepts as described in the [Objectification Thesaurus](https://github.com/husky-helen/MObyGaze/blob/main/dataset/objectification-thesaurus.json)
* `comment`: annotator's comment on annotations
* `start_frame`, `end_frame`: start and end frame of the annotations
* `imdb_key`: IMDb key of the movie
* `movie_title`: title of the movie
* `framerate`: movie framerate
* `text`: subtitles contained within that segment (defined by start and end frame)
* `label_speech`: 1 if a Speech concept is present, 0 otherwise
* `start_clip`, `end_clip`: indicate the start and end frames within 16-frame segments. For example, if an annotation spans from frame 0 to frame 32, start_clip would be 0 and end_clip would be 1.
* `label_audio`: 1 if an Audio concept is present, 0 otherwise





