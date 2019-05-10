/////////////////////////////////////////////////////////////////////////////
// Name:        thumbnailctrl.h
// Purpose:     Displays a scrolling window of thumbnails
// Author:      Julian Smart
// Modified by: Anil Kumar
// Created:     03/08/04 17:22:46
// RCS-ID:      
// Copyright:   (c) Julian Smart
// Licence:     wxWidgets Licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_THUMBNAILCTRL_H_
#define _WX_THUMBNAILCTRL_H_

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "thumbnailctrl.cpp"
#endif

#include "wx/dynarray.h"

/*! Styles
 */

#define wxTH_MULTIPLE_SELECT    0x0010
#define wxTH_SINGLE_SELECT      0x0000
#define wxTH_TEXT_LABEL         0x0020
#define wxTH_IMAGE_LABEL        0x0040

/*! Flags
 */

#define wxTHUMBNAIL_SHIFT_DOWN  0x01
#define wxTHUMBNAIL_CTRL_DOWN   0x02
#define wxTHUMBNAIL_ALT_DOWN    0x04

/*! Defaults
 */

#define wxTHUMBNAIL_DEFAULT_OVERALL_SIZE wxSize(-1, -1)
#define wxTHUMBNAIL_DEFAULT_IMAGE_SIZE wxSize(80, 80)
#define wxTHUMBNAIL_DEFAULT_SPACING 6
#define wxTHUMBNAIL_DEFAULT_MARGIN 3
#define wxTHUMBNAIL_DEFAULT_UNFOCUSSED_BACKGROUND wxColour(175, 175, 175)
#define wxTHUMBNAIL_DEFAULT_FOCUSSED_BACKGROUND wxColour(140, 140, 140)
#define wxTHUMBNAIL_DEFAULT_UNSELECTED_BACKGROUND wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE)
#define wxTHUMBNAIL_DEFAULT_TYPE_COLOUR wxColour(0, 0, 200)
#define wxTHUMBNAIL_DEFAULT_TAG_COLOUR wxColour(0, 0, 255)
#define wxTHUMBNAIL_DEFAULT_FOCUS_RECT_COLOUR wxColour(100, 80, 80)

class wxThumbnailCtrl;

// Drawing styles/states
#define wxTHUMBNAIL_SELECTED    0x01
#define wxTHUMBNAIL_TAGGED      0x02
// The control is focussed
#define wxTHUMBNAIL_FOCUSSED    0x04
// The item itself has the focus
#define wxTHUMBNAIL_IS_FOCUS    0x08
#define wxTHUMBNAIL_IS_HOVER    0x10

class wxThumbnailItem : public wxObject {
DECLARE_CLASS(wxThumbnailItem)

public:
    explicit wxThumbnailItem(const wxString &label = wxEmptyString) : m_label(label), m_state(0) {}

    /// Label
    void SetLabel(const wxString &filename) {
        m_label = filename;
        m_bitmap = wxNullBitmap;
    }

    const wxString &GetLabel() const { return m_label; }

    /// State storage while sorting
    void SetState(int state) { m_state = state; }

    int GetState() const { return m_state; }

    /// Refresh the item
    virtual bool Refresh(wxThumbnailCtrl *ctrl, int index);

    /// Draw the background
    virtual bool
    DrawBackground(wxDC &dc, wxThumbnailCtrl *ctrl, const wxRect &rect, const wxRect &imageRect, int style, int index);

    /// Draw the item
    virtual bool Draw(wxDC &dc, wxThumbnailCtrl *ctrl, const wxRect &rect, int style, int index);

    wxBitmap &GetBitmap() { return m_bitmap; }

private:
    wxBitmap m_bitmap;
    wxString m_label;
    int m_state; // state storage while sorting
};

WX_DECLARE_OBJARRAY(wxThumbnailItem, wxThumbnailItemArray);

class wxThumbnailCtrl : public wxScrolledCanvas {
DECLARE_CLASS(wxThumbnailCtrl)

DECLARE_EVENT_TABLE()

public:
    wxThumbnailCtrl();

    explicit wxThumbnailCtrl(wxWindow *parent, wxWindowID id = -1, const wxPoint &pos = wxDefaultPosition,
                             const wxSize &size = wxDefaultSize,
                             long style = wxTH_TEXT_LABEL | wxTH_IMAGE_LABEL | wxBORDER_THEME);

    /// Creation
    bool Create(wxWindow *parent, wxWindowID id = -1, const wxPoint &pos = wxDefaultPosition,
                const wxSize &size = wxDefaultSize,
                long style = wxTH_TEXT_LABEL | wxTH_IMAGE_LABEL | wxBORDER_THEME);

    /// Scrolls the item into view if necessary
    void EnsureVisible(int n);

    /// Sorts items
    void Sort();

    /// Show the tooltip
    virtual void ShowTooltip(int n);

    /// Append a single item
    virtual int Append(wxThumbnailItem *item);

    /// Insert a single item
    virtual int Insert(wxThumbnailItem *item, int pos = 0);

    /// Clear all items
    virtual void Clear();

    /// Delete this item
    virtual void Delete(int n);

    /// Get the number of items in the control
    virtual int GetCount() const { return m_items.GetCount(); }

    /// Is the control empty?
    bool IsEmpty() const { return GetCount() == 0; }

    /// Get the nth item
    wxThumbnailItem *GetItem(int n);

    /// Get the overall rect of the given item
    /// If transform is true, rect is relative to the scroll viewport
    /// (i.e. may be negative)
    bool GetItemRect(int item, wxRect &rect, bool transform = true);

    /// Get the image rect of the given item
    bool GetItemRectImage(int item, wxRect &rect, bool transform = true);

    /// Return the row and column given the client
    /// size and a left-to-right, top-to-bottom layout
    /// assumption
    bool GetRowCol(int item, const wxSize &clientSize, int &row, int &col);

    /// Get the focus item, or -1 if there is none
    int GetFocusItem() const { return m_focusItem; }

    /// Set the focus item
    void SetFocusItem(int item);

    /// Select or deselect an item
    void Select(int n, bool select = true);

    /// Select or deselect a range
    void SelectRange(int from, int to, bool select = true);

    /// Tag or untag an item
    void Tag(int n, bool tag = true);

    /// Select all
    void SelectAll();

    /// Select none
    void SelectNone();

    /// Get the index of the single selection, if not multi-select.
    /// Returns -1 if there is no selection.
    int GetSelection() const;

    /// Get indexes of all selections, if multi-select
    const wxArrayInt &GetSelections() const { return m_selections; }

    /// Get indexes of all tags
    const wxArrayInt &GetTags() const { return m_tags; }

    /// Returns true if the item is selected
    bool IsSelected(int n) const;

    /// Returns true if the item is tagged
    bool IsTagged(int n) const;

    /// Clears all selections
    void ClearSelections();

    /// Clears all tags
    void ClearTags();

    /// Get mouse hover item
    int GetMouseHoverItem() const { return m_hoverItem; }

    /// Find the item under the given point
    bool HitTest(const wxPoint &pt, int &n);

    /// The overall size of the thumbnail, including decorations.
    /// DON'T USE THIS from the application, since it will
    /// normally be calculated by SetThumbnailImageSize.
    void SetThumbnailOverallSize(const wxSize &sz) { m_thumbnailOverallSize = sz; }

    const wxSize &GetThumbnailOverallSize() const { return m_thumbnailOverallSize; }

    /// The size of the image part
    void SetThumbnailImageSize(const wxSize &sz);

    const wxSize &GetThumbnailImageSize() const { return m_thumbnailImageSize; }

    /// The inter-item spacing
    void SetSpacing(int spacing) { m_spacing = spacing; }

    int GetSpacing() const { return m_spacing; }

    /// The margin between elements within the thumbnail
    void SetThumbnailMargin(int margin) { m_thumbnailMargin = margin; }

    int GetThumbnailMargin() const { return m_thumbnailMargin; }

    /// The height required for text in the thumbnail
    void SetThumbnailTextHeight(int h) { m_thumbnailTextHeight = h; }

    int GetThumbnailTextHeight() const { return m_thumbnailTextHeight; }

    /// Get tag bitmap
    const wxBitmap &GetTagBitmap() const { return m_tagBitmap; }

    /// The focussed and unfocussed background colour for a
    /// selected thumbnail
    void SetSelectedThumbnailBackgroundColour(const wxColour &focussedColour, const wxColour &unfocussedColour) {
        m_focussedThumbnailBackgroundColour = focussedColour;
        m_unfocussedThumbnailBackgroundColour = unfocussedColour;
    }

    const wxColour &GetSelectedThumbnailFocussedBackgroundColour() const { return m_focussedThumbnailBackgroundColour; }

    const wxColour &
    GetSelectedThumbnailUnfocussedBackgroundColour() const { return m_unfocussedThumbnailBackgroundColour; }

    /// The unselected background colour for a thumbnail
    void
    SetUnselectedThumbnailBackgroundColour(const wxColour &colour) { m_unselectedThumbnailBackgroundColour = colour; }

    const wxColour &GetUnselectedThumbnailBackgroundColour() const { return m_unselectedThumbnailBackgroundColour; }

    /// The colour for the type text (top left of thumbnail)
    void SetTypeColour(const wxColour &colour) { m_typeColour = colour; }

    const wxColour &GetTypeColour() const { return m_typeColour; }

    /// The colour for the tag outline
    void SetTagColour(const wxColour &colour) { m_tagColour = colour; }

    const wxColour &GetTagColour() const { return m_tagColour; }

    /// The focus rectangle pen colour
    void SetFocusRectColour(const wxColour &colour) { m_focusRectColour = colour; }

    const wxColour &GetFocusRectColour() const { return m_focusRectColour; }

    /// The thumbnail outlines show or not
    void ShowOutlines(bool flag = true) { m_showOutlines = flag; }

    bool IsOutlinesShown() const { return m_showOutlines; }

    /// Painting
    void OnDraw(wxDC &dc) override;

protected:
    /// Command handlers
    void OnSelectAll(wxCommandEvent &event);

    void OnUpdateSelectAll(wxUpdateUIEvent &event);

    /// Mouse-events
    void OnMouse(wxMouseEvent &event);

    /// Left-click-down
    void OnLeftClickDown(wxMouseEvent &event);

    /// Left-click-up
    void OnLeftClickUp(wxMouseEvent &event);

    /// Left-double-click
    void OnLeftDClick(wxMouseEvent &event);

    /// Mouse-motion
    void OnMouseMotion(wxMouseEvent &event);

    /// Mouse-leave
    void OnMouseLeave(wxMouseEvent &event);

    /// Right-click-down
    void OnRightClickDown(wxMouseEvent &event);

    /// Right-click-up
    void OnRightClickUp(wxMouseEvent &event);

    /// Key press
    void OnChar(wxKeyEvent &event);

    /// Sizing
    void OnSize(wxSizeEvent &event);

    /// Setting/losing focus
    void OnSetFocus(wxFocusEvent &event);

    void OnKillFocus(wxFocusEvent &event);

    // Implementation

    /// Draws the item
    bool DrawItem(int n, wxDC &dc, const wxRect &rect, const wxRect &imageRect, int style);

    void SetMouseHoverItem(int n, int flags = 0);

    /// Set up scrollbars, e.g. after a resize
    void SetupScrollbars();

    /// Calculate the outer thumbnail size based
    /// on font used for text and inner size
    void CalculateOverallThumbnailSize();

    /// Do (de)selection
    void DoSelection(int n, int flags);

    /// Keyboard navigation
    virtual bool Navigate(int keyCode, int flags);

    /// Scroll to see the image
    void ScrollIntoView(int n, int keyCode);

    /// Paint the background
    void PaintBackground(wxDC &dc);

    /// Thumbnails compare function
    virtual int Compare(wxThumbnailItem **item1, wxThumbnailItem **item2);

private:
    /// Member initialisation
    void Init();

    /// The items
    wxThumbnailItemArray m_items;

    /// The selections
    wxArrayInt m_selections;

    /// The tags
    wxArrayInt m_tags;

    /// Outer size of the thumbnail item
    wxSize m_thumbnailOverallSize;

    /// Image size of the thumbnail item
    wxSize m_thumbnailImageSize;

    /// The inter-item spacing
    int m_spacing;

    /// The margin between the image/text and the edge of the thumbnail
    int m_thumbnailMargin;

    /// The height of thumbnail text in the current font
    int m_thumbnailTextHeight;

    /// First selection in a range
    int m_firstSelection;

    /// Last selection
    int m_lastSelection;

    /// Focus item
    int m_focusItem;

    /// Tag marker bitmap
    wxBitmap m_tagBitmap;

    /// Outlines flag
    bool m_showOutlines;

    /// Mouse hover item
    int m_hoverItem = wxNOT_FOUND;

    /// Current control, used in sorting
    static wxThumbnailCtrl *sm_currentThumbnailCtrl;

    /// Focussed/unfocussed selected thumbnail background colours
    wxColour m_focussedThumbnailBackgroundColour;
    wxColour m_unfocussedThumbnailBackgroundColour;
    wxColour m_unselectedThumbnailBackgroundColour;
    wxColour m_focusRectColour;

    /// Type text colour
    wxColour m_typeColour;

    /// Tag colour
    wxColour m_tagColour;

    /// Drag start position
    wxPoint m_dragStartPosition = wxDefaultPosition;
};

/*!
 * wxThumbnailEvent - the event class for wxThumbnailCtrl notifications
 */

class wxThumbnailEvent : public wxNotifyEvent {
public:
    wxThumbnailEvent(wxEventType commandType = wxEVT_NULL, int winid = 0)
            : wxNotifyEvent(commandType, winid),
              m_itemIndex(-1), m_flags(0) {}

    wxThumbnailEvent(const wxThumbnailEvent &event)
            : wxNotifyEvent(event),
              m_itemIndex(event.m_itemIndex), m_flags(event.m_flags) {}

    int GetIndex() const { return m_itemIndex; }

    void SetIndex(int n) { m_itemIndex = n; }

    const wxArrayInt &GetItemsIndex() const { return m_itemsIndex; }

    void SetItemsIndex(const wxArrayInt &itemsIndex) { m_itemsIndex = itemsIndex; }

    int GetFlags() const { return m_flags; }

    void SetFlags(int flags) { m_flags = flags; }

    const wxPoint &GetPosition() const { return m_position; }

    void SetPosition(const wxPoint &position) { m_position = position; }

    virtual wxEvent *Clone() const { return new wxThumbnailEvent(*this); }

protected:
    int m_itemIndex;
    int m_flags;
    wxPoint m_position;
    wxArrayInt m_itemsIndex;

private:
DECLARE_DYNAMIC_CLASS_NO_ASSIGN(wxThumbnailEvent)
};

/*!
 * wxThumbnailCtrl event macros
 */
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_SELECTION_CHANGED, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_ITEM_SELECTED, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_ITEM_DESELECTED, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_LEFT_CLICK, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_RIGHT_CLICK, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_VIEW_RIGHT_CLICK, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_LEFT_DCLICK, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_RETURN, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_DRAG_START, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_SORTED, wxThumbnailEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_THUMBNAIL_ITEM_HOVER_CHANGED, wxThumbnailEvent);

#endif
// _WX_THUMBNAILCTRL_H_
