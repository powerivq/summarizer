package summarizer_test

import (
	"testing"

	"github.com/powerivq/summarizer"
)

func TestPruneInvisibleCharacters(t *testing.T) {
	str := "测试：	中\u200d文\n\nTest\n\t  123"
	pruned := summarizer.PruneInvisibleCharacters(str)
	if pruned != "测试：	中文\n\nTest\n\t  123" {
		t.Errorf("PruneInvisibleCharacters() = %v", pruned)
	}
}
