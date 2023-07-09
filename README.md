# Python Text Sorter

A common use case I have run into with large outputs from ChatGPT is overlapping topics in the chunked lists it generates. Manually sorting and condensing these hundreds of lines is tedious, so I created this text sorter to group similar lines of text, making the process much quicker.

## Iterations

There are 3 iterations of this sorting that I kept mostly because the different algorithms were all new to me so I like having the reference to each type.

For almost all cases </textSorter_DBSCAN.py> is the correct choice.

### Initial - LDA

The first sorting algorithm was a Latent Dirichlet Allocation. It was able to group some obvious similarities but then would also leave other line pairs with multiple shared keywords separated.

### KMeans and Agglomerative Clustering

Next I tried a mix of KMeans (to help quantify the number of clusters) and Agglomerative Clustering to group the items. This performed better but still seemed to be missing a few obvious similarities. The similarity checking algorithm did improve during this round though.

### DBSCAN

With the similarity better quantified in the second iteration it made sense to move to a DBSCAN algorithm. This performs very well for keyword matching and still factors in the overall line similarity to some extent. This was good enough to satisfy my needs.

At this level I also added tests with mock data so that I could quickly validate if the code would work for other datasets beyond the one I currently had available. Looks good enough to me!

## Future Improvements

I could add another level of sorting so that the clusters are grouped by similarity as well. This would help for some edge cases where a single line item sort of fits with two otherwise distinct clusters, ensuring those clusters are near each other in the final list.

I could also add a summarization of sorts for each cluster in an attempt to complete the condensing as well. I want to run a few data sets through this process manually first so that I can more easily tell how automatable that is (ie does a cluster usually become 1 topic or multiple?). This could also be an avenue to bring in the GPT APIs if necessary.
