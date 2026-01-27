// Package segment provides index building and search capabilities.
package segment

import (
	"strings"
	"unicode"

	"github.com/tidepool/tidepool/internal/document"
)

// Index represents a simple inverted index for a segment.
type Index struct {
	// TermToDocIDs maps normalized terms to document IDs
	TermToDocIDs map[string][]string `json:"term_to_doc_ids"`

	// DocIDToTerms maps document IDs to their terms (for scoring)
	DocIDToTerms map[string][]string `json:"doc_id_to_terms"`

	// DocIDToTitle maps document IDs to their titles
	DocIDToTitle map[string]string `json:"doc_id_to_title"`

	// TagsIndex maps tags to document IDs
	TagsIndex map[string][]string `json:"tags_index"`

	// TotalDocs is the total number of documents in the segment
	TotalDocs int `json:"total_docs"`
}

// BuildIndex builds an inverted index from documents.
func BuildIndex(docs []*document.Document) *Index {
	idx := &Index{
		TermToDocIDs: make(map[string][]string),
		DocIDToTerms: make(map[string][]string),
		DocIDToTitle: make(map[string]string),
		TagsIndex:    make(map[string][]string),
		TotalDocs:    len(docs),
	}

	for _, doc := range docs {
		// Index content
		terms := Tokenize(doc.Content)
		if doc.Title != "" {
			titleTerms := Tokenize(doc.Title)
			terms = append(terms, titleTerms...)
		}

		// Deduplicate terms for this document
		termSet := make(map[string]struct{})
		for _, term := range terms {
			termSet[term] = struct{}{}
		}

		var uniqueTerms []string
		for term := range termSet {
			uniqueTerms = append(uniqueTerms, term)
			idx.TermToDocIDs[term] = append(idx.TermToDocIDs[term], doc.ID)
		}

		idx.DocIDToTerms[doc.ID] = uniqueTerms
		idx.DocIDToTitle[doc.ID] = doc.Title

		// Index tags
		for _, tag := range doc.Tags {
			normalizedTag := strings.ToLower(tag)
			idx.TagsIndex[normalizedTag] = append(idx.TagsIndex[normalizedTag], doc.ID)
		}
	}

	return idx
}

// Tokenize splits text into normalized tokens.
func Tokenize(text string) []string {
	// Convert to lowercase and split on non-alphanumeric characters
	var tokens []string
	var currentToken strings.Builder

	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentToken.WriteRune(r)
		} else {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
		}
	}

	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}

	// Filter out very short tokens (stopwords would go here in a real system)
	var filtered []string
	for _, token := range tokens {
		if len(token) >= 2 {
			filtered = append(filtered, token)
		}
	}

	return filtered
}

// Search performs a simple search over the index.
func (idx *Index) Search(query string, limit int) []string {
	queryTerms := Tokenize(query)
	if len(queryTerms) == 0 {
		return nil
	}

	// Count how many query terms match each document
	docScores := make(map[string]int)

	for _, term := range queryTerms {
		if docIDs, ok := idx.TermToDocIDs[term]; ok {
			for _, docID := range docIDs {
				docScores[docID]++
			}
		}
	}

	// Convert to sorted slice
	type scoredDoc struct {
		ID    string
		Score int
	}

	var scoredDocs []scoredDoc
	for docID, score := range docScores {
		scoredDocs = append(scoredDocs, scoredDoc{ID: docID, Score: score})
	}

	// Sort by score descending
	for i := 0; i < len(scoredDocs); i++ {
		for j := i + 1; j < len(scoredDocs); j++ {
			if scoredDocs[j].Score > scoredDocs[i].Score {
				scoredDocs[i], scoredDocs[j] = scoredDocs[j], scoredDocs[i]
			}
		}
	}

	// Take top results
	if limit > 0 && len(scoredDocs) > limit {
		scoredDocs = scoredDocs[:limit]
	}

	result := make([]string, len(scoredDocs))
	for i, sd := range scoredDocs {
		result[i] = sd.ID
	}

	return result
}

// FilterByTags returns document IDs that have all the specified tags.
func (idx *Index) FilterByTags(tags []string) []string {
	if len(tags) == 0 {
		return nil
	}

	// Start with documents matching the first tag
	normalizedFirst := strings.ToLower(tags[0])
	docIDs, ok := idx.TagsIndex[normalizedFirst]
	if !ok {
		return nil
	}

	// Intersect with documents matching remaining tags
	docSet := make(map[string]struct{})
	for _, id := range docIDs {
		docSet[id] = struct{}{}
	}

	for i := 1; i < len(tags); i++ {
		normalizedTag := strings.ToLower(tags[i])
		tagDocIDs, ok := idx.TagsIndex[normalizedTag]
		if !ok {
			return nil
		}

		tagDocSet := make(map[string]struct{})
		for _, id := range tagDocIDs {
			tagDocSet[id] = struct{}{}
		}

		// Intersect
		for id := range docSet {
			if _, exists := tagDocSet[id]; !exists {
				delete(docSet, id)
			}
		}
	}

	result := make([]string, 0, len(docSet))
	for id := range docSet {
		result = append(result, id)
	}
	return result
}
